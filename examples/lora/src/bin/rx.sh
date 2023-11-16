#!/bin/bash

BUILD_TYPE=""
#BUILD_TYPE="--release"
CENTER_FREQ="--center-freq=867000000"
RX_OFFSET="--rx-freq-offset=1.1e6"
#TX_OFFSET="--tx-freq-offset=1.1e6"
DEVICE_FILTER="--device-filter=driver=aaronia_http,tx_url=http://172.18.0.1:54665,url=http://172.18.0.1:54664"
RX_GAIN="--rx-gain 10"
#TX_DEVICE_FILTER="--device-filter=driver=soapy,soapy_driver=uhd,type=b200,name=B200mini"
#TX_GAIN="--tx-gain 40"
RX_ANTENNA=""
#TX_ANTENNA=""
SAMPLE_RATE="--sample-rate 125000"
SPREADING_FACTOR="--spreading-factor 12"
BANDWIDTH="--bandwidth 125000"

export FUTURESDR_LOG_LEVEL=debug
export RUST_BACKTRACE=full

cargo run --bin rx ${BUILD_TYPE} -- ${SPREADING_FACTOR} ${BANDWIDTH} ${CENTER_FREQ} ${RX_OFFSET} ${DEVICE_FILTER} ${RX_GAIN} ${RX_ANTENNA} ${SAMPLE_RATE}
