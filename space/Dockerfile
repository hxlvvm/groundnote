# Hugging Face Space (docker SDK): the stdlib web app from groundnote/app.py, CPU only.
FROM python:3.12-slim
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers==5.17.0
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
COPY --chown=user groundnote groundnote
EXPOSE 7860
CMD ["python", "-m", "groundnote.app", "Opflare1144/groundnote-97m", "--host", "0.0.0.0", "--port", "7860"]
