// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersrc.h>
#include <libavfilter/buffersink.h>
#include <libavutil/opt.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifndef HAVE_ASPRINTF
#include <stdarg.h>

int asprintf(char **ret_str, const char *format, ...) {
    va_list args, args2;
    va_start(args, format);
    va_copy(args2, args);
    int len = vsnprintf(NULL, 0, format, args2) + 1;
    char *str = malloc(len);
    int ret = vsnprintf(str, len, format, args);
    *ret_str = str;
    return ret;
}
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef char *(*buffer_cb)(int16_t *data, size_t size, void *userdata);

static char *format_error(const char *outer, int err) {
    char *str_err = NULL;
    char inner[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(err, inner, AV_ERROR_MAX_STRING_SIZE);
    asprintf(&str_err, "%s: %s", outer, inner);
    return str_err;
}

struct filter {
    AVFilterGraph *graph;
    AVFilterContext *src_ctx;
    AVFilterContext *sink_ctx;
};

struct buffer {
    int16_t *data;
    size_t size;
    size_t ptr;
};

static char *add_to_buffer(AVFrame *frame, struct buffer *buffer, buffer_cb callback, void *userdata) {
    int16_t *data = (int16_t *)frame->data[0];
    for (int i = 0; i < frame->nb_samples; i++) {
        buffer->data[buffer->ptr++] = data[i];
        if (buffer->ptr == buffer->size) {
            char *err = callback(buffer->data, buffer->ptr, userdata);
            if (err) {
                return err;
            }
            buffer->ptr = 0;
        }
    }
    return NULL;
}

static char *receive_and_filter(AVFrame *frame, AVFrame *filt_frame, AVCodecContext *dec_ctx, struct filter *filter, struct buffer *buffer, buffer_cb callback, void *userdata) {
    int err;
    char *str_err = NULL;
    while ((err = avcodec_receive_frame(dec_ctx, frame)) == 0) {
        err = av_buffersrc_add_frame_flags(filter->src_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
        if (err < 0) {
            str_err = format_error("adding frame to filter", err);
            goto loop_end;
        }
        while ((err = av_buffersink_get_frame(filter->sink_ctx, filt_frame)) == 0) {
            str_err = add_to_buffer(filt_frame, buffer, callback, userdata);
            av_frame_unref(filt_frame);
            if (str_err != NULL) {
                goto loop_end;
            }
        }
        if (err != AVERROR_EOF && err != AVERROR(EAGAIN)) {
            str_err = format_error("receiving frame from filter", err);
        }
      loop_end:
        av_frame_unref(frame);
        if (str_err != NULL) {
            return str_err;
        }
    }
    if (err != AVERROR_EOF && err != AVERROR(EAGAIN)) {
        str_err = format_error("receiving frame from decoder", err);
    }
    return str_err;
}

static struct filter create_filter_graph(AVRational time_base, AVCodecContext *dec_ctx, char **error) {
    char *str_err = NULL;
    AVFilterGraph *graph = avfilter_graph_alloc();
    AVFilterInOut *filter_in = avfilter_inout_alloc();
    AVFilterInOut *filter_out = avfilter_inout_alloc();
    const AVFilter *src_filt = avfilter_get_by_name("abuffer");
    AVFilterContext *filt_src_ctx = NULL;
    AVFilterContext *filt_sink_ctx = NULL;
    char *abuffer_args;
    char ch_layout[64];
    int err = av_channel_layout_describe(&dec_ctx->ch_layout, ch_layout, 64);
    if (err < 0) {
        str_err = format_error("getting channel layout", err);
        goto err_free_graph;
    }
    asprintf(&abuffer_args, "time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%s",
             time_base.num, time_base.den, dec_ctx->sample_rate,
             av_get_sample_fmt_name(dec_ctx->sample_fmt), ch_layout);
    err = avfilter_graph_create_filter(&filt_src_ctx, src_filt, "in", abuffer_args, NULL, graph);
    free(abuffer_args);
    if (err < 0) {
        str_err = format_error("creating input filter", err);
        goto err_free_graph;
    }
    const AVFilter *sink_filt = avfilter_get_by_name("abuffersink");
    err = avfilter_graph_create_filter(&filt_sink_ctx, sink_filt, "out",
                                       NULL, NULL, graph);
    if (err < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot create audio buffer sink\n");
        goto err_free_graph;
    }

    enum AVSampleFormat sample_fmts[2] = { AV_SAMPLE_FMT_S16, -1 };
    err = av_opt_set_int_list(filt_sink_ctx, "sample_fmts", sample_fmts, -1,
                              AV_OPT_SEARCH_CHILDREN);
    if (err < 0) {
        str_err = format_error("setting sample format", err);
        goto err_free_graph;
    }

    err = av_opt_set(filt_sink_ctx, "ch_layouts", "mono", AV_OPT_SEARCH_CHILDREN);
    if (err < 0) {
        str_err = format_error("setting channel layout", err);
        goto err_free_graph;
    }

    int sample_rates[2] = { 16000, -1 };
    err = av_opt_set_int_list(filt_sink_ctx, "sample_rates", sample_rates, -1,
                              AV_OPT_SEARCH_CHILDREN);
    if (err < 0) {
        str_err = format_error("setting sample rate", err);
        goto err_free_graph;
    }

    filter_out->name       = av_strdup("in");
    filter_out->filter_ctx = filt_src_ctx;
    filter_out->pad_idx    = 0;
    filter_out->next       = NULL;

    filter_in->name       = av_strdup("out");
    filter_in->filter_ctx = filt_sink_ctx;
    filter_in->pad_idx    = 0;
    filter_in->next       = NULL;

    err = avfilter_graph_parse_ptr(graph, "aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono",
                                   &filter_in, &filter_out, NULL);
    if (err < 0) {
        str_err = format_error("parsing filter description", err);
        goto err_free_graph;
    }
    err = avfilter_graph_config(graph, NULL);
    if (err < 0) {
        str_err = format_error("configuring filter graph", err);
        goto err_free_graph;
    }

  out:
    avfilter_inout_free(&filter_in);
    avfilter_inout_free(&filter_out);
    *error = str_err;
    return (struct filter){
        .src_ctx = filt_src_ctx,
        .graph = graph,
        .sink_ctx = filt_sink_ctx,
    };
  err_free_graph:
    avfilter_graph_free(&graph);
    goto out;
}

static int write_packet(struct buffer *write_buf, const uint8_t *buf, int buf_size) {
    if (write_buf->ptr + buf_size > write_buf->size) {
        size_t nsize = write_buf->size * 2;
        write_buf->data = realloc(write_buf->data, nsize);
        write_buf->size = nsize;
    }
    char *target = (char*)write_buf->data;
    memcpy(target + write_buf->ptr, buf, buf_size);
    write_buf->ptr += buf_size;
    return 0;
}

char *encode_audio(int16_t *buf, size_t size, buffer_cb py_callback) {
    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_OPUS);
    if (!codec) {
        return strdup("Unable to find codec for opus");
    }
    AVFormatContext *out_ctx;
    int err = avformat_alloc_output_context2(&out_ctx, NULL, "ogg", NULL);
    if (err < 0) {
        return strdup("Unable to create format context");
    }
    AVStream *stream = avformat_new_stream(out_ctx, NULL);
    if (!stream) {
        return strdup("Failed to create stream");
    }
    AVCodecContext *encode_ctx = avcodec_alloc_context3(codec);
    encode_ctx->bit_rate = 32000;
    encode_ctx->sample_fmt = AV_SAMPLE_FMT_S16;
    encode_ctx->sample_rate = 16000;
    encode_ctx->ch_layout = (AVChannelLayout)AV_CHANNEL_LAYOUT_MONO;
    char *str_err = NULL;
    err = avcodec_open2(encode_ctx, codec, NULL);
    if (err < 0) {
        str_err = format_error("opening decoder", err);
        goto out_free_ctx;
    }
    err = avcodec_parameters_from_context(stream->codecpar, encode_ctx);
    if (err < 0) {
        str_err = format_error("filling codec parameters", err);
        goto out_free_ctx;
    }
    stream->time_base = encode_ctx->time_base;
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    frame->nb_samples = encode_ctx->frame_size;
    frame->format = encode_ctx->sample_fmt;
    frame->pts = 0;
    err = av_channel_layout_copy(&frame->ch_layout, &encode_ctx->ch_layout);
    if (err < 0) {
        str_err = format_error("copying layout", err);
        goto out_free_frame;
    }
    err = av_frame_get_buffer(frame, 0);
    if (err < 0) {
        str_err = format_error("allocating buffers", err);
        goto out_free_frame;
    }
    void *io_buf = av_malloc(4096);
    struct buffer write_buf = {
        .data = malloc(1024 * 1024),
        .ptr = 0,
        .size = 1024 * 1024
    };
    out_ctx->pb = avio_alloc_context(io_buf, 4096, 1, &write_buf, NULL, (void*)write_packet, NULL);
    if (!out_ctx->pb) {
        str_err = strdup("Failed allocating io context");
        goto out_free_bufs;
    }
    err = avformat_write_header(out_ctx, NULL);
    if (err < 0) {
        str_err = format_error("writing file headers", err);
        goto out_free_io_ctx;
    }
    size_t processed = 0;
    while (processed < size) {
        err = av_frame_make_writable(frame);
        if (err < 0) {
            str_err = format_error("making frame writeable", err);
            goto out_free_io_ctx;
        }
        int16_t *samples = (int16_t*)frame->data[0];
        size_t to_process = MIN(frame->nb_samples, size - processed);
        memcpy(samples, buf + processed, to_process * sizeof(int16_t));
        processed += to_process;
        frame->nb_samples = (int)to_process;
        err = avcodec_send_frame(encode_ctx, frame);
        if (err < 0) {
            str_err = format_error("sending packet to encoder", err);
            goto out_free_io_ctx;
        }
        while ((err = avcodec_receive_packet(encode_ctx, packet)) == 0) {
            err = av_interleaved_write_frame(out_ctx, packet);
            if (err < 0) {
                str_err = format_error("writing packet", err);
                goto out_free_io_ctx;
            }
            av_packet_unref(packet);
        }
        if (err != AVERROR_EOF && err != AVERROR(EAGAIN)) {
            str_err = format_error("receiving packet", err);
            goto out_free_io_ctx;
        }
        AVRational pts_off = av_mul_q(av_div_q(encode_ctx->time_base, stream->time_base), av_make_q((int)to_process, 1));
        frame->pts += (int64_t) av_q2d(pts_off);
    }
    err = avcodec_send_frame(encode_ctx, NULL);
    if (err < 0) {
        str_err = format_error("flushing encoder", err);
        goto out_free_io_ctx;
    }

    while ((err = avcodec_receive_packet(encode_ctx, packet)) == 0) {
        err = av_interleaved_write_frame(out_ctx, packet);
        if (err < 0) {
            str_err = format_error("writing packet", err);
            goto out_free_io_ctx;
        }
        av_packet_unref(packet);
    }
    if (err != AVERROR_EOF && err != AVERROR(EAGAIN)) {
        str_err = format_error("receiving packet", err);
        goto out_free_io_ctx;
    }

    err = av_write_trailer(out_ctx);
    if (err < 0) {
        str_err = format_error("writing trailer", err);
        goto out_free_io_ctx;
    }
    str_err = py_callback(write_buf.data, write_buf.ptr, NULL);
  out_free_io_ctx:
    avio_context_free(&out_ctx->pb);
  out_free_bufs:
    av_free(io_buf);
    free(write_buf.data);
  out_free_frame:
    av_frame_free(&frame);
    av_packet_free(&packet);
  out_free_ctx:
    avcodec_free_context(&encode_ctx);
    avformat_free_context(out_ctx);
    return str_err;
}

static char *find_audio_stream(const AVFormatContext *fmt_ctx, int *stream_id) {
    *stream_id = -1;
    for (int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            *stream_id = i;
            break;
        }
    }
    if (*stream_id == -1) {
        return strdup("Unable to find the audio stream");
    }
    return NULL;
}

EXPORT char *get_duration(const char *path, uint64_t *duration) {
    AVFormatContext *fmt_ctx = NULL;
    char *str_err = NULL;
    int err = avformat_open_input(&fmt_ctx, path, NULL, NULL);
    if (err < 0) {
        str_err = format_error("opening input", err);
        goto out_ret;
    }
    err = avformat_find_stream_info(fmt_ctx, NULL);
    if (err < 0) {
        str_err = format_error("getting stream info", err);
        goto out_close_input;
    }
    *duration = (uint64_t)(fmt_ctx->duration / AV_TIME_BASE);

  out_close_input:
    avformat_close_input(&fmt_ctx);
  out_ret:
    return str_err;
}

EXPORT char *decode_audio(const char *path, size_t block_size, buffer_cb callback, void *userdata) {
    AVFormatContext *fmt_ctx = NULL;
    char *str_err = NULL;
    int err = avformat_open_input(&fmt_ctx, path, NULL, NULL);
    if (err < 0) {
        str_err = format_error("opening input", err);
        goto out_ret;
    }
    err = avformat_find_stream_info(fmt_ctx, NULL);
    if (err < 0) {
        str_err = format_error("getting stream info", err);
        goto out_close_input;
    }

    int stream_id;
    str_err = find_audio_stream(fmt_ctx, &stream_id);
    if (str_err != NULL) {
        goto out_close_input;
    }

    AVCodecParameters *codec_params = fmt_ctx->streams[stream_id]->codecpar;
    const AVCodec *decoder = avcodec_find_decoder(codec_params->codec_id);
    if (!decoder) {
        asprintf(&str_err, "can't find decoder for codec: %s", avcodec_get_name(codec_params->codec_id));
        goto out_close_input;
    }
    AVCodecContext *dec_ctx = avcodec_alloc_context3(decoder);
    err = avcodec_parameters_to_context(dec_ctx, codec_params);
    if (err < 0) {
        str_err = format_error("adding codec parameters", err);
        goto out_free_dec_ctx;
    }
    err = avcodec_open2(dec_ctx, decoder, NULL);
    if (err < 0) {
        str_err = format_error("opening decoder", err);
        goto out_free_dec_ctx;
    }
    char *graph_err;
    struct filter filter = create_filter_graph(fmt_ctx->streams[stream_id]->time_base, dec_ctx, &graph_err);
    if (graph_err != NULL) {
        asprintf(&str_err, "creating filter graph: %s", graph_err);
        free(graph_err);
        goto out_free_dec_ctx;
    }

    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    AVFrame *filt_frame = av_frame_alloc();
    struct buffer buffer = {
        .data = malloc(block_size * sizeof(float)),
        .ptr = 0,
        .size = block_size
    };
    while ((err = av_read_frame(fmt_ctx, packet)) == 0) {
        if (packet->stream_index != stream_id) {
            continue;
        }
        err = avcodec_send_packet(dec_ctx, packet);
        if (err < 0) {
            str_err = format_error("sending packet to decoder", err);
            goto loop_end;
        }
        str_err = receive_and_filter(frame, filt_frame, dec_ctx, &filter, &buffer, callback, userdata);
      loop_end:
        av_packet_unref(packet);
        if (str_err != NULL) {
            goto out_free_packet;
        }
    }
    if (err != AVERROR_EOF) {
        str_err = format_error("reading frame", err);
        goto out_free_packet;
    }
    err = avcodec_send_packet(dec_ctx, NULL);
    if (err < 0) {
        str_err = format_error("sending packet to decoder", err);
        goto out_free_packet;
    }
    str_err = receive_and_filter(frame, filt_frame, dec_ctx, &filter, &buffer, callback, userdata);
    if (str_err == NULL) {
        str_err = callback(buffer.data, buffer.ptr, userdata);
    }

  out_free_packet:
    free(buffer.data);
    av_frame_free(&frame);
    av_frame_free(&filt_frame);
    av_packet_free(&packet);
    avfilter_graph_free(&filter.graph);
  out_free_dec_ctx:
    avcodec_free_context(&dec_ctx);
  out_close_input:
    avformat_close_input(&fmt_ctx);
  out_ret:
    return str_err;
}

EXPORT char *transcode_audio(const char *path, size_t block_size, buffer_cb callback) {
    return decode_audio(path, block_size, (void*)encode_audio, callback);
}
