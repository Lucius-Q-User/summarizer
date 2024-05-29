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
    int len = _vscprintf(format, args2) + 1;
    char *str = malloc(len);
    int ret = vsnprintf(str, len, format, args);
    *ret_str = str;
    return ret;
}
#endif

typedef int (*buffer_cb)(float *data, size_t size);

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
    float *data;
    size_t size;
    size_t ptr;
};

static char *add_to_buffer(AVFrame *frame, struct buffer *buffer, buffer_cb callback) {
    int16_t *data = (int16_t *)frame->data[0];
    for (int i = 0; i < frame->nb_samples; i++) {
        float sample = (float)(data[i]) / 32768.0f;
        buffer->data[buffer->ptr++] = sample;
        if (buffer->ptr == buffer->size) {
            if (callback(buffer->data, buffer->ptr) < 0) {
                return strdup("callback error");
            }
            buffer->ptr = 0;
        }
    }
    return NULL;
}

static char *receive_and_filter(AVFrame *frame, AVFrame *filt_frame, AVCodecContext *dec_ctx, struct filter *filter, struct buffer *buffer, buffer_cb callback) {
    int err;
    char *str_err = NULL;
    while ((err = avcodec_receive_frame(dec_ctx, frame)) == 0) {
        err = av_buffersrc_add_frame_flags(filter->src_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF);
        if (err < 0) {
            str_err = format_error("adding frame to filter", err);
            goto loop_end;
        }
        while ((err = av_buffersink_get_frame(filter->sink_ctx, filt_frame)) == 0) {
            str_err = add_to_buffer(filt_frame, buffer, callback);
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

EXPORT char *decode_audio(const char *path, size_t block_size, buffer_cb callback) {
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

    int stream_id = -1;
    for (int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            stream_id = i;
            break;
        }
    }
    if (stream_id == -1) {
        str_err = strdup("Unable to find the audio stream");
        goto out_close_input;
    }

    AVCodecParameters *codec_params = fmt_ctx->streams[stream_id]->codecpar;
    AVCodec *decoder = avcodec_find_decoder(codec_params->codec_id);
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
        str_err = receive_and_filter(frame, filt_frame, dec_ctx, &filter, &buffer, callback);
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
    str_err = receive_and_filter(frame, filt_frame, dec_ctx, &filter, &buffer, callback);
    if (str_err == NULL) {
        if (callback(buffer.data, buffer.ptr) < 0) {
            str_err = strdup("callback error");
        }
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
