#!/bin/bash

#BUILD_TYPE="--release"
BUILD_TYPE=""
CENTER_FREQ="--center-freq=867000000"
RX_OFFSET="--rx-freq-offset=1.1e6"
#DEVICE_FILTER="--device-filter=driver=soapy,soapy_driver=uhd,type=b200,name=B200mini"
#RX_GAIN="--rx-gain 60"
DEVICE_FILTER="--device-filter=driver=aaronia_http,tx_url=http://172.18.0.1:54665,url=http://172.18.0.1:54664"
RX_GAIN="--rx-gain 10"
RX_ANTENNA=""
SAMPLE_RATE="--sample-rate 125000"

# find port: ss -u
# nc -u 10.193.0.73 [port] -p 18570 [on novo 3]

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
cd /usr/local/src/FutureSDR/examples/lora || cd ${SCRIPTPATH}/FutureSDR/examples/lora || echo asdf

export FUTURESDR_LOG_LEVEL=debug
export RUST_BACKTRACE=full
#export FUTURESDR_LOG_LEVEL=warn
#export FUTURESDR_CTRLPORT_ENABLE=true
#export FUTURESDR_CTRLPORT_BIND="0.0.0.0:1348"

cargo run --bin rx ${BUILD_TYPE} -- ${CENTER_FREQ} ${RX_OFFSET} ${DEVICE_FILTER} ${RX_GAIN} ${RX_ANTENNA} ${SAMPLE_RATE}
