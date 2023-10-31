#!/bin/bash

BUILD_TYPE="--release"
CENTER_FREQ="--center-freq=2.45e9"
RX_OFFSET="--rx-freq-offset=-4e6"
DEVICE_FILTER="--device-filter=driver=soapy,soapy_driver=uhd,type=b200,name=B200mini"
RX_GAIN="--rx-gain 60"
RX_ANTENNA=""
SAMPLE_RATE="--sample-rate 4e6"

# find port: ss -u
# nc -u 10.193.0.73 [port] -p 18570 [on novo 3]

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
cd /usr/local/src/FutureSDR/examples/lora || cd ${SCRIPTPATH}/FutureSDR/examples/lora || echo asdf

export FUTURESDR_LOG_LEVEL=debug
#export FUTURESDR_LOG_LEVEL=warn
#export FUTURESDR_CTRLPORT_ENABLE=true
#export FUTURESDR_CTRLPORT_BIND="0.0.0.0:1348"

cargo run --bin rx ${BUILD_TYPE} -- ${CENTER_FREQ} ${RX_OFFSET} ${DEVICE_FILTER} ${RX_GAIN} ${RX_ANTENNA} ${SAMPLE_RATE}
