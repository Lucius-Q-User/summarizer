## An LLM-powered youtube video summarization tool.

Uses a locally running copy of Mistral 7b to generate per-section summaries of youtube videos. The summary is based either on the automatic google captions, or if those are not available - on the transcript generated using whisper on local machine.

### Installation
```
pip install llama-summarizer
```

### Basic usage
```
summarize https://www.youtube.com/watch?v=dQw4w9WgXcQ
```
Once completed, the summary will be opened in the system default browser.

### Sponsorblock integration
```
summarize -sb sponsor -sb selfpromo -sb intro https://www.youtube.com/watch?v=TuHOf_kZK6Q
```
Excludes the sections marked as Sponsor, Self Promotion and Intermission from the summary.
