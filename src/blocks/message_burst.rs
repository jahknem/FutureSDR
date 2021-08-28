use anyhow::Result;

use crate::runtime::AsyncKernel;
use crate::runtime::Block;
use crate::runtime::BlockMeta;
use crate::runtime::BlockMetaBuilder;
use crate::runtime::MessageIo;
use crate::runtime::MessageIoBuilder;
use crate::runtime::Pmt;
use crate::runtime::StreamIo;
use crate::runtime::StreamIoBuilder;
use crate::runtime::WorkIo;

pub struct MessageBurst {
    message: Pmt,
    n_messages: u64,
}

impl MessageBurst {
    pub fn new(message: Pmt, n_messages: u64) -> Block {
        Block::new_async(
            BlockMetaBuilder::new("MessageBurst").build(),
            StreamIoBuilder::new().build(),
            MessageIoBuilder::new().register_output("out").build(),
            MessageBurst {
                message,
                n_messages,
            },
        )
    }
}

#[async_trait]
impl AsyncKernel for MessageBurst {
    async fn work(
        &mut self,
        io: &mut WorkIo,
        _sio: &mut StreamIo,
        mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
    ) -> Result<()> {
        for _ in 0..self.n_messages {
            mio.post(0, self.message.clone()).await;
        }

        io.finished = true;
        Ok(())
    }
}

pub struct MessageBurstBuilder {
    message: Pmt,
    n_messages: u64,
}

impl MessageBurstBuilder {
    pub fn new(message: Pmt, n_messages: u64) -> MessageBurstBuilder {
        MessageBurstBuilder {
            message,
            n_messages,
        }
    }

    pub fn build(self) -> Block {
        MessageBurst::new(self.message, self.n_messages)
    }
}