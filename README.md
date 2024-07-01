# llava-next-playground

>[!NOTE]
>
>Use `pip install flash-attn --no-build-isolation` to install flash-attn. 
> 
> Rye does not support the option "--no-build-isolation."
> See [the issue](https://github.com/astral-sh/rye/issues/552).

## Setup

Install dependencies
```bash
pip install .
```

or
```bash
rye sync
```

After install using rye or pip, install flash-attn separately
```bash
pip install flash-attn --no-build-isolation
```
or
```bash
uv pip install flash-attn --no-build-isolation
```

## Usage

To run with a sample video (data/sample.mp4) and using 15 frames
```bash
rye run main
```

You can specify the video file and number of frames and enter your own prompt
```bash
echo "Describe this video." | rye run main --file-path data/sample.mp4 --num-frames 15
```
