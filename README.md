# 🎬 Auto Subtitle & Translate with Whisper + LLaMA

Automatically generate subtitles for any video using [OpenAI Whisper](https://openai.com/blog/whisper), overlay them with `ffmpeg`, and optionally translate them to 50+ languages using a [LLaMA2-based multilingual translator](https://huggingface.co/SnypzZz/Llama2-13b-Language-translate).

### 📺 [Demo Video](https://youtu.be/vkvTpmQ7M48?si=qQLvYzwtsQ4djo4K)

---

## 🛠️ Installation

Make sure you have Python 3.7 or later.

Install the package directly from GitHub:

```bash
pip install git+https://github.com/YJ-20/auto-subtitle-llama
```

### Install `ffmpeg`

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

---

## 🚀 Usage

### Only Transcribe (w/o Translate)

```bash
auto_subtitle_llama /path/to/video.mp4
```

### Transcribe and Translate Subtitles

To translate subtitles to another language:

```bash
auto_subtitle_llama /path/to/video.mp4 --translate_to ko_KR
```

---

## 🌐 Supported Translation Languages

| Language     | Code   | Language     | Code   | Language     | Code   |
|--------------|--------|--------------|--------|--------------|--------|
| Arabic       | ar_AR  | Czech        | cs_CZ  | German       | de_DE  |
| English      | en_XX  | Spanish      | es_XX  | Estonian     | et_EE  |
| Finnish      | fi_FI  | French       | fr_XX  | Gujarati     | gu_IN  |
| Hindi        | hi_IN  | Italian      | it_IT  | Japanese     | ja_XX  |
| Kazakh       | kk_KZ  | Korean       | ko_KR  | Lithuanian   | lt_LT  |
| Latvian      | lv_LV  | Burmese      | my_MM  | Nepali       | ne_NP  |
| Dutch        | nl_XX  | Romanian     | ro_RO  | Russian      | ru_RU  |
| Sinhala      | si_LK  | Turkish      | tr_TR  | Vietnamese   | vi_VN  |
| Chinese      | zh_CN  | Afrikaans    | af_ZA  | Azerbaijani  | az_AZ  |
| Bengali      | bn_IN  | Persian      | fa_IR  | Hebrew       | he_IL  |
| Croatian     | hr_HR  | Indonesian   | id_ID  | Georgian     | ka_GE  |
| Khmer        | km_KH  | Macedonian   | mk_MK  | Malayalam    | ml_IN  |
| Mongolian    | mn_MN  | Marathi      | mr_IN  | Polish       | pl_PL  |
| Pashto       | ps_AF  | Portuguese   | pt_XX  | Swedish      | sv_SE  |
| Swahili      | sw_KE  | Tamil        | ta_IN  | Telugu       | te_IN  |
| Thai         | th_TH  | Tagalog      | tl_XX  | Ukrainian    | uk_UA  |
| Urdu         | ur_PK  | Xhosa        | xh_ZA  | Galician     | gl_ES  |
| Slovene      | sl_SI  |              |        |              |        |


---

## 📦 Other Options

| Option             | Description |
|--------------------|-------------|
| `--model`          | Default: `turbo`. Whisper (transcribing) model size. |
| `--output_dir, -o` | Default: `subtitled/`. Directory where the resulting subtitled videos and `.srt` files will be saved. |
| `--srt_only`       | Default: `false`. If set to `true`, only the `.srt` subtitle file will be generated without creating a subtitled video. Useful for manual subtitle editing or external video processing pipelines. |

#### Example:

```bash
# Choose Whisper model size
auto_subtitle_llama /path/to/video.mp4 --model medium

# Save output to a custom directory
auto_subtitle_llama /path/to/video.mp4 --output_dir results/

# Generate only .srt file (no video overlay)
auto_subtitle_llama /path/to/video.mp4 --srt_only true
```

Available whisper models:
```
tiny, base, small, medium, large, turbo
```

---

## 📘 Command-line Help

To view all available options:

```bash
auto_subtitle_llama --help
```

---

## ⚖️ License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for more details.
#   w i s 2 
 
 