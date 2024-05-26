FROM python:3.11

ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"

RUN git clone https://github.com/yjoer/bank-account-fraud.git
WORKDIR /bank-account-fraud
COPY .env .

RUN curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash
RUN rye sync

CMD ["rye", "run", "python", "bench.py"]
