use async_net::{TcpListener, TcpStream};
use futures::AsyncReadExt;
use futures::AsyncWriteExt;
use futuresdr::log::{info, debug};

use futuresdr::anyhow::{bail, Context, Result};
use futuresdr::async_trait::async_trait;
use futuresdr::runtime::Block;
use futuresdr::runtime::BlockMeta;
use futuresdr::runtime::BlockMetaBuilder;
use futuresdr::runtime::Kernel;
use futuresdr::runtime::MessageIo;
use futuresdr::runtime::MessageIoBuilder;
use futuresdr::runtime::StreamIo;
use futuresdr::runtime::StreamIoBuilder;
use futuresdr::runtime::WorkIo;

const TCP_EXCHANGER_PORT: u32 = 1592;

/// Push samples into a TCP socket.
pub struct TcpExchanger {
    local_ip: String,
    remote_ip: String,
    socket: Option<TcpStream>,
}

impl TcpExchanger {
    pub fn new(local_ip: String, remote_ip: String) -> Block {
        Block::new(
            BlockMetaBuilder::new("TcpSource").build(),
            StreamIoBuilder::new().add_input::<u8>("in").add_output::<u8>("out").build(),
            MessageIoBuilder::new().build(),
            TcpExchanger {
                local_ip,
                remote_ip,
                socket: None,
            },
        )
    }
}

#[doc(hidden)]
#[async_trait]
impl Kernel for TcpExchanger {
    async fn work(
        &mut self,
        io: &mut WorkIo,
        sio: &mut StreamIo,
        _mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
    ) -> Result<()> {

        let out = sio.output(0).slice::<u8>();
        if out.is_empty() {
            return Ok(());
        }

        match self
            .socket
            .as_mut()
            .context("no socket")?
            .read_exact(out)
            .await
        {
            Ok(_) => {
                debug!("tcp source read bytes {}", out.len());
                sio.output(0).produce(out.len());
            }
            Err(_) => {
                debug!("tcp source socket closed");
                io.finished = true;
            }
        }

        let i = sio.input(0).slice::<u8>();

        match self
            .socket
            .as_mut()
            .context("no socket")?
            .write_all(i)
            .await
        {
            Ok(()) => {}
            Err(_) => bail!("tcp sink socket error"),
        }

        if sio.input(0).finished() {
            io.finished = true;
        }

        debug!("tcp sink wrote bytes {}", i.len());
        sio.input(0).consume(i.len());

        Ok(())
    }

    async fn init(
        &mut self,
        _sio: &mut StreamIo,
        _mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
    ) -> Result<()> {
        if self.local_ip < self.remote_ip {
            info!("acting as tcp exchanger server");
            let mut listener = Some(TcpListener::bind(format!("{}:{}", self.local_ip, TCP_EXCHANGER_PORT)).await?);
            let (socket, _) = listener
                .as_mut()
                .context("no listener")?
                .accept()
                .await?;
            self.socket = Some(socket);
            info!("remote tcp exchanger accepted connection");
        }
        else {
            info!("acting as tcp exchanger client");
            self.socket = Some(TcpStream::connect(format!("{}:{}", self.remote_ip, TCP_EXCHANGER_PORT)).await?);
            info!("connected remote tcp exchanger");
        }
        Ok(())
    }
}
