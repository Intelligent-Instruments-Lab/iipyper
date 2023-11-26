from typing import Tuple, Any
import time
import json
from threading import Thread
import numpy as np

from pythonosc import osc_packet
# from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.osc_server import BlockingOSCUDPServer, ThreadingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient

from .state import _lock

__all__ = ['OSC', 'ndarray_to_json', 'ndarray_from_json', 'osc_blob_encode', 'osc_blob_decode', 'ndarray_from_osc_args', 'ndarray_to_osc_args']

# leaving this here for now. seems like it may not be useful since nested bundles
# do not appear to work in sclang.
# class BundleDispatcher(Dispatcher):
#     def call_handlers_for_packet(self,
#              data: bytes, client_address: Tuple[str, int]
#              ) -> None:
#         """Override python-osc to handle whole bundles in one callback.
#             data: Data of packet
#             client_address: Address of client this packet originated from
#         """
#         # Get OSC messages from all bundles or standalone message.
#         try:
#             packet = osc_packet.OscPacket(data)
#             addrs = [msg.message.address for msg in packet.messages]
#             # get common root address
#             root_addr = '/root' # TODO
#             stem_addrs = addrs # TODO
#             handler = self.handlers_for_address(root_addr)
#             # TODO: handle time
#             # times = [msg.time for msg in packet.messages]
#             fused_message = []
#             for stem, msg in zip(stem_addrs, packet.messages):
#                 fused_message.append(stem)
#                 fused_message.append(msg.message)
#             handler.invoke(client_address, fused_message)
#         except osc_packet.ParseError:
#             pass

def do_json(d, k, json_keys, route):
    v = d[k]
    if not isinstance(v, str): return
    has_prefix = v.startswith('%JSON:')
    if has_prefix:
        v = v[6:]
    if k in json_keys or has_prefix:
        try:
            data = json.loads(v)
            if all(key in data for key in ('dtype','shape','data')):
                # assume numpy array
                data = ndarray_from_json(v)
            d[k] = data
        except (TypeError, json.JSONDecodeError) as e:
            print(f"""
            warning: JSON decode failed for {route} argument "{k}": 
            value: {v}
            {type(e)} {e}
            """)

def ndarray_to_json(array: np.ndarray) -> str:
    array_list = array.tolist()
    array_info = {
        'data': array_list,
        'dtype': str(array.dtype),
        'shape': array.shape
    }
    return json.dumps(array_info)

def ndarray_from_json(json_str: str) -> np.ndarray:
    data = json.loads(json_str)
    array_data = data['data']
    dtype = data['dtype']
    shape = data['shape']
    return np.array(array_data, dtype=dtype).reshape(shape)

def osc_blob_encode(data_bytes: bytes) -> bytes:
    """Encode an OSC-compliant blob from bytes data."""
    blob_size = len(data_bytes)
    blob_size_packed = blob_size.to_bytes(4, byteorder='big')
    padding_length = (4 - blob_size % 4) % 4
    padding = b'\x00' * padding_length
    return blob_size_packed + data_bytes + padding

def osc_blob_decode(osc_blob: bytes) -> bytes:
    """Decode data out of an OSC blob.

    OSC blob format:
        size: 32-bit big-endian integer (first 4 bytes).
        data: binary blob data.
        padding: zero-padding to make total size multiple of 4.
    """
    try:
        if not isinstance(osc_blob, bytes) or len(osc_blob) < 4:
            raise ValueError("OSC blob must be a bytes object with ≥4 bytes.")
        size = int.from_bytes(osc_blob[:4], 'big')
        if len(osc_blob) < 4 + size:
            raise ValueError(f"OSC blob length ({len(osc_blob)}) < reported size {size}.")
        return osc_blob[4:4+size]
    except Exception as e:
        raise ValueError(f"Error parsing OSC blob ({osc_blob}): {e}.")

def ndarray_from_osc_args(*args) -> np.ndarray:
    """Parse OSC arguments into an ndarray.
    
    Args:
        args: Variable length argument list, expected to contain
            dtype (str), dimensions (int), and binary-encoded data (bytes).

    Example:
        >>> ndarray_args('float32', 100, 32, <12800 byte blob>)

    Returns:
        np.frombuffer(blob, dtype=dtype).reshape(shape)
    """
    if len(args) < 3:
        raise ValueError(f"Minimum 3 args required; dtype(str), shape(int), data(bytes), got {len(args)}.")

    args = args[1:] # skip 'ndarray' arg
    dtype = args[0]
    shape = args[1:-1]
    blob = osc_blob_decode(args[-1])

    if not isinstance(dtype, str):
        raise ValueError(f"`dtype` must be a str, got {type(dtype)}")
    if not all(isinstance(dim, int) for dim in shape):
        raise ValueError(f"`shape` dims must be ints, got {shape}.")

    try:
        return np.frombuffer(blob, dtype=dtype).reshape(shape)
    except ValueError as e:
        raise ValueError(f"Couldn't parse args. Error: {e}")

def ndarray_to_osc_args(arr: np.ndarray) -> Tuple[Any, ...]:
    """Encode an ndarray into OSC arguments."""
    dtype = arr.dtype.name
    shape = arr.shape
    data = osc_blob_encode(arr.tobytes())
    return ('ndarray', dtype, *shape, data)

class OSC():
    """
    TODO: Handshake between server and clients
    TODO: Polling clients after handshake
    TODO: Enqueuing and buffering messages (?)
    """
    def __init__(self, host="127.0.0.1", port=9999, verbose=True,
         concurrent=False):
        """
        TODO: Expand to support multiple IPs + ports

        Args:
            host (str): IP address
            port (int): port to receive on
            verbose (bool): whether to print activity
            concurrent (bool): if True, handle each incoming OSC message on 
                its own thread. otherwise, incoming OSC is handled serially on 
                one thread for the whole OSC object.
        """
        self.verbose = verbose
        self.concurrent = concurrent
        self.host = host
        self.port = port
        self.dispatcher = Dispatcher()
        self.server = None
        self.clients = {} # (host,port) -> client
        self.client_names = {} # (name) -> (host,port)

        self.create_server()

    def create_server(self):#, host=None, port=None):
        """
        Create the server
        """
        # if (host is None):
        #     host = self.host
        # if (port is None):
        #     port = self.port
        cls = ThreadingOSCUDPServer if self.concurrent else BlockingOSCUDPServer

        if (self.server is None):
            self.server = cls((self.host, self.port), self.dispatcher)
            if self.verbose:
                print(f"OSC server created {self.host}:{self.port}")

            # start the OSC server on its own thread
            Thread(target=self.server.serve_forever, daemon=True).start()
            # self.server.serve_forever()
        else:
            print("OSC server already exists")

    # def close_server(self):
    #     """
    #     Close the server
    #     """
    #     if (self.server is not None):
    #         self.transport.close()
    #     else:
    #         print("OSC server does not exist")

    def add_handler(self, address, handler):
        """
        Map the custom message handler to the OSC dispatcher
        """
        # if (self.server is not None):
        self.dispatcher.map(address, handler, needs_reply_address=True)

    def create_client(self, name, host=None, port=None):
        """
        Add an OSC client.
        Args:
            name: name this client
            host (int): IP to send to, defaults to same as server
            port (int): port to send to, defaults to 57120 (supercollider)
        """
        if (host == None):
            host = self.host
        if (port == None):
            port = 57120
        if ((host, port) not in self.clients):
            self.clients[host, port] = SimpleUDPClient(host, port)
            if self.verbose:
                print(f"OSC client created {host}:{port}")
        else:
            print("OSC client already exists")
        self.client_names[name] = (host, port)

    def get_client_by_name(self, name):
        try:
            return self.clients[self.client_names[name]]
        except Exception:
            print(f'no client with name "{name}"')
            return None

    def get_client_by_sender(self, address):
        if address not in self.clients:
            host, port = address
            self.create_client(f'{host}:{port}', host, port)
        return self.clients[address]

    def send(self, route, *msg, client=None):
        """
        Send message to default client, or with client in address

        Args:
            address: '/my/osc/route' or 'host:port/my/osc/route'
            *msg: content
            client: name of client or None
        """
        if client is not None:
            client = self.get_client_by_name(client)
        elif ':' in route:
            try:
                client_str, route = route.split('/', 1)
                assert ':' in client_str
                host, port = client_str.split(':')
                assert '/' not in host
                port = int(port)
                client = self.get_client_by_sender((host, port))
            except Exception:
                print(f'failed to get client address from OSC route "{route}"')
        else:
            client = next(iter(self.clients.values()))

        if client is None:
            print(f'OSC message failed to send, could not determine client')
            return

        if not route.startswith('/'):
            route = '/'+route

        # handle numpy arrays
        if len(msg) > 1 and isinstance(msg[0], np.ndarray):
            raise ValueError(f"Found len(msg)>1 & type np.ndarray. Only 1 ndarray arg can be sent per message.")
        elif len(msg) == 1 and isinstance(msg[0], np.ndarray):
            msg = ndarray_to_osc_args(msg[0])
        client.send_message(route, msg)

        if self.verbose:
            print(f"OSC message sent {route}:{msg}")

    def _decorate(self, use_kwargs, route, return_host, return_port, json_keys):
        """generic decorator (args and kwargs cases)"""
        if hasattr(route, '__call__'):
            # bare decorator
            f = route
            route = None
            json_keys = set()
        else:
            f = None
            json_keys = set(json_keys or [])

        def decorator(f, route=route, 
                return_host=return_host, return_port=return_port, 
                json_keys=json_keys):
            # default_route = f'/{f.__name__}/*'
            if route is None:
                route = f'/{f.__name__}'
            # print(route)
            assert isinstance(route, str) and route.startswith('/')

            def handler(client, address, *args):
                """
                Args:
                    client: (host,port) of sender
                    address: full OSC address
                    *args: content of OSC message
                """
                # print('handler:', client, address)
                if use_kwargs:
                    kwargs = {k:v for k,v in zip(args[::2], args[1::2])}
                    # JSON conversions
                    for k in kwargs: 
                        do_json(kwargs, k, json_keys, route)
                    args = []
                else:
                    kwargs = {}

                with _lock:
                    # handle numpy arrays
                    if len(args) > 0:
                        if args[0] == 'ndarray':
                            args = ndarray_from_osc_args(*args)
                            print(args)
                            ret = f(address, args)
                    else:
                        ret = f(address, *args, **kwargs)
                if ret is not None:
                    self.return_to_sender_by_sender(ret, client, return_host, return_port)

            self.add_handler(route, handler)

            return f

        return decorator if f is None else decorator(f)
    
    def return_to_sender_by_sender(self, return_to: tuple, sender: tuple, return_host=None, return_port=None):
        '''
        Args:
            return_to: tuple of (route, *args) to send back to sender
            client: (host,port) of sender
            return_host: host to send back to, defaults to sender
            return_port: port to send back to, defaults to sender
        
        Send return value as message back to sender by client(host,port).
        '''
        if not hasattr(return_to, '__len__'):
            print(f"""
            value returned from OSC handler should start with route, got: {return_to}
            """)
        else:
            sender = (
                sender[0] if return_host is None else return_host,
                sender[1] if return_port is None else return_port
            )
            self.get_client_by_sender(sender).send_message(return_to[0], return_to[1:])

    def return_to_sender_by_name(self, return_to: tuple, client_name: str):
        '''
        Args:
            return_to: tuple of (route, *args) to send back to sender
            client_name: name of client
        
        Send return value as message back to sender by name.
        '''
        if not hasattr(return_to, '__len__'):
            print(f"""
            value returned from OSC handler should start with route, got: {return_to}
            """)
        else:
            self.get_client_by_name(client_name).send_message(return_to[0], return_to[1:])

    def args(self, route=None, return_host=None, return_port=None):
        """decorate a function as an args-style OSC handler.

        the decorated function should look like:
        def f(route, my_arg, my_arg2, ...):
            ...
        the OSC message will be converted to python types and passed as positional
        arguments.
        """
        return self._decorate(False, route, return_host, return_port, None)

    def kwargs(self, route=None, return_host=None, return_port=None, json_keys=None):
        """decorate a function as an kwargs-style OSC handler.

        the decorated function should look like:
        def f(route, my_key=my_value, ...):
            ...
        the incoming OSC message should alternate argument names with values:
            /osc/route 'my_key' value 'my_key2' value ...
        
        Args:
            route: specify the OSC route. if None, use the function name
            json_keys: names of keyword arguments which should be decoded
                from JSON to python objects, 
                in the case that they arrive as strings.
                alternatively, if a string starts with '%JSON:' it will be decoded.
        """
        return self._decorate(True, route, return_host, return_port, json_keys)

    def ndarray(self, route=None, dformat:str=None, return_host=None, return_port=None):
        """decorate a function as an ndarray OSC handler.

        Args:
            route: full OSC address
            dformat: required data format of the ndarray (expects 'bytes' or 'json').
                    bytes expects args:
                        'ndarray', 'float32', 100, 32, <12800 byte blob>
                    json expects kwargs:
                        'ndarray': json_str

        the decorated function should look like:
        def f(route, ndarray):
            ...
        the OSC message will be parsed into an ndarray.
        """
        if dformat is None:
            raise ValueError("Data format not specified (expected 'bytes'/'json').")
        elif not isinstance(dformat, str):
            raise ValueError(f"Data format should be str, got {type(dformat)}.")
        elif dformat == 'bytes':
            return self.args(route, return_host, return_port)
        elif dformat == 'json':
            return self.kwargs(route, return_host, return_port, ['ndarray'])
        else:
            raise ValueError(f"Invalid data format '{dformat}' (expected 'bytes'/'json').")

    def __call__(self, client, *a, **kw):
        """alternate syntax for `send` with client name first"""
        self.send(*a, client=client, **kw)
