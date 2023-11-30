import time
import json
import pickle
import inspect
import typing
from threading import Thread

# from pythonosc import osc_packet
# from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.osc_server import BlockingOSCUDPServer, ThreadingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient

import pydantic

from .types import *
from .util import maybe_lock

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

def _consume_splat(hd:Any, tl:Iterable, cls:type, is_key) -> Tuple[Any, Any]:
    params = typing.get_args(cls)
    splat_items = []

    if len(params)==0:
        # print(cls)
        # read until a string or end of iteration is encountered
        while hd is not None and not is_key(hd):
            splat_items.append(hd)
            hd = next(tl, None)
        return splat_items, hd
    
    elif len(params)==1:
        # print(cls)
        # read the annotated number of items
        for _ in range(params[0]):
            if hd is None:
                raise ValueError(f"""
                hit end of arguments while parsing {cls}
                """)
            splat_items.append(hd)
            hd = next(tl, None)
        return splat_items, hd

    else:
        raise TypeError
    
def _is_legacy_json_str(item):
    if isinstance(item, str) and item.startswith('%JSON:'):
        return item[6:], True
    else:
        return item, False

def _consume_items(hd:Any, tl:Iterable, cls:type, is_key) -> Tuple[Any, Any]:
    """
    Args:
        hd: first element of items
        tl: remaining items (an iterator)
        cls: annotated type to consume
        is_key: test if an item indicates an argument name
    Returns:
        instance of cls constructed from consumed items,
        next item following those consumed (or None if end of iteration)
    """
    # consume groups of items annotated as Vector
    if hasattr(cls, '__name__') and cls.__name__=='Splat':
        return _consume_splat(hd, tl, cls, is_key)
    
    hd, is_legacy_json = _is_legacy_json_str(hd)

    if cls is object or is_legacy_json:
        # interpret object-anotated strings as JSON
        if isinstance(hd, str):
            hd = json.loads(hd)
        # interpret object-annotated blobs as pickled objects
        elif isinstance(hd, bytes):
            # pickle
            hd = pickle.loads(hd)

    if cls is NDArray:
        # interpret NDArrray-anotated strings as iipyper JSON or numpy repr format
        if isinstance(hd, str):
            if hd.startswith('array'):
                hd = ndarray_from_repr(hd)
            else:
                hd = ndarray_from_json(hd)
        # interpret NDArray-annotated blobs as raw float32 buffers
        elif isinstance(hd, bytes):
            hd = np.frombuffer(hd, dtype=np.float32)

    #single item case: return the head, advance to first element of tail
    return hd, next(tl, None)

def _parse_osc_items(osc_items, sig_info, kwargs) -> Tuple[List, Dict[str, Any]]:
    """
    convert a list of osc message contents to positional and keyword arguments
    of t
    """
    positional_params, named_params, has_varp, has_varkw = sig_info

    position = 0 if len(positional_params) or has_varp else -1

    items = iter(osc_items)
    args = []
    kw = {}

    # decide if an item represents the name of an argument,
    # or is a positional argument or part of a Splat[None]
    def is_key(item):
        # print(f"""
        #     {kwargs=} 
        #     and {isinstance(item, str)=} 
        #     and ({(item in named_params)=} or (
        #         {has_varkw=} and {(position<0)=}
        #         ))
        #       """)
        return (
            kwargs # named arguments enabled
            and isinstance(item, str) # can be a name
            and (item in named_params # is a known name
                or (has_varkw and position<0)) # or must be a ** arg
        )
    try:
        _, item = _consume_items(None, items, None, is_key)
        while item is not None:
            # print(item, is_key(item))
            if is_key(item):
                ### interpret this item as the name of an argument
                key = item
                # print(f'{key=}')
                if item in named_params:
                    cls = named_params[item].annotation
                else:
                    cls = Any
                try:
                    value, item = _consume_items(
                        next(items), items, cls, is_key)
                except json.JSONDecodeError:
                    print(f'JSON error decoding argument "{key}"')
                    raise
                kw[key] = value
                position = -1 # no more positional arguments allowed
            elif position>=0:
                ### interpret this item as a positional argument
                # print(f'{position=}')
                if position < len(positional_params):
                    cls = positional_params[position].annotation
                elif has_varp:
                    cls = Any
                else:
                    raise ValueError("""
                    too many positional arguments.
                    """)
                if (
                    cls is Splat[None] 
                    and not kwargs 
                    and position<len(positional_params)-1
                    ):
                    raise ValueError("""
                    Splat[None] can only be used on the last argument when kwargs=False.
                    """)
                try:
                    value, item = _consume_items(
                        item, items, cls, is_key)
                except json.JSONDecodeError:
                    print('JSON error decoding argument', position)
                    raise
                args.append(value)
                position += 1
                # check if this was the final positional argument
                if position >= len(positional_params) and not has_varp:
                    position = -1
            else:
                raise ValueError("""
                positional argument after keyword arg while parsing OSC
                """)
    except StopIteration:
        pass
    return args, kw


class OSC():
    """
    TODO: Handshake between server and clients
    TODO: Polling clients after handshake
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
        client.send_message(route, msg)
        if self.verbose:
            print(f"OSC message sent {route}:{msg}")
    
    def handle(self, 
            route:str=None, return_host:str=None, return_port:int=None,
            kwargs=True, lock=True):
        """
        OSC handler decorator supporting mixed args and kwargs, typing.

        The decorated function will receive the OSC route as its first argument.
        Further arguments will be the elements of the OSC message, which can be
        converted in various ways by supplying type annotations (see below).

        If the decorated function returns a value, it should be a tuple beginning
        with the OSC route to reply to, followed by the message contents.

        Args:
            route: OSC path for this handler. If not given,
                use the name of the decorated function.
                If this is a callable, assume the decorator is being used 'bare'
                with all arguments set to default values, i.e. `osc.handle(f)`
                is equivalent to `osc.handle()(f)`.
            return_host: hostname of reply address for return value.
                if not given, reply to sender.
            return_port: port of reply address for return value.
                if not given, reply to sender port.
                NOTE: replying on the same port seems to work with SuperCollider,
                but not with Max.
            kwargs: if True (default), parse OSC message for key-value pairs
                corresponding to named arguments of the decorated function.
            lock: if True (default), use the global iipyper lock around the
                decorated function

        keyword arguments of the decorated function:
            if a string with the same name as a parameter is found, 
            the following item in the OSC message will be used as the value.
            positional arguments can still be used before any keyword pairs,
            like in Python.

            Example:
            ```
            @osc.handle
            def my_func(route, a, b, c=2, d=3):
                print(f'{a=}, {b=}, {c=}, {d=}')
            ```
            OSC message: /my_func/ 0 1 2 'd' 3
            prints: "a=0, b=1, c=2, d=3"

            This is idiomatic in SuperCollider, where flat key, value pairs are
            seen, e.g. `~synth.set('freq', 440, 'amp', 0.2)`
            It's also often easier than managing nested data structures in Max/Pd.

            However, there can be ambiguity when strings are positional arguments.
            It's recommended to avoid using strings as positional arguments, 
            or else to ensure that there will be no collision with the names
            of arguments to the decorated function, 
            for example by adding _ to your argument names.
            you can also disable the keyword argument parsing with `kwargs=False`.

        type annotations:
            object:
                decode a string as JSON
                decode bytes through pickle

            Splat[N]: 
                consume N consecutive items from the OSC message items into a 
                tuple, which forms one argument to the decorated function.

                @osc.handle
                f(a:Splat[2], b:Splat[3]) 
                OSC Message: /f 0 banana 2 3 4.5
                a = (0, banana)
                b = (2, 3, 4.5) 

            Splat[None]: 
                consume items up to the next keyword item into a list
                NOTE: generally should not be used when there are strings 
                in the OSC message, and must be used as a keyword argument, 
                since a str matching an argument name delimits the end of the
                Splat (unless the Splat is the final argument).

                f(a:Splat[None]) -- fine, collects all arguments into a list
                f(a:Splat[None], b:Splat[None]=None) -- breaks if `a` should 
                contain a string `'b'`.
                    OSC message: /f a 0 1 2 b 3 4
                    would call `f('/f', [0,1,2], [3,4])`.

            NDArray: 
                decode a string (either JSON or repr format) to a numpy array
                    JSON format: see `iipyper.ndarray_to_json`
                        (no support for complex dtypes)
                    repr format: see `numpy.array_repr`
                decode bytes to a 1-dimensional float32 array
                    TODO: support annotating dtype via pydantic_numpy types

            other types:
                will be decoded by python-osc and validated by pydantic

        """
        if hasattr(route, '__call__'):
            # bare decorator
            f = route
            route = None
        else:
            f = None

        def decorator(f, route=route, 
                return_host=return_host, return_port=return_port):
            # default_route = f'/{f.__name__}/*'
            if route is None:
                route = f'/{f.__name__}'
            # print(route)
            assert isinstance(route, str) and route.startswith('/')

            # get info out of function signature
            sig = inspect.signature(f)

            positional_params = []
            named_params = {}
            has_varp = False
            has_varkw = False
            pos = True
            for i,(name,p) in enumerate(sig.parameters.items()):
                if i==0:
                    # skip first argument (the OSC route)
                    continue
                if p.kind in (p.VAR_KEYWORD, p.VAR_KEYWORD):
                    pos = False
                else:
                    named_params[name] = p
                if pos:
                    positional_params.append(p)
                has_varp |= p.kind==p.VAR_POSITIONAL
                has_varkw |= p.kind==p.VAR_KEYWORD

            if has_varkw and not kwargs:
                raise ValueError(f"""
                ERROR: iipyper: OSC handler {f} was created with kwargs=False,
                but the decorated function has a ** argument
                """)

            # wrap with pydantic validation decorator
            f = pydantic.validate_call(f)

            def handler(client, address, *osc_items):
                """
                Args:
                    client: (host,port) of sender
                    address: full OSC address
                    *args: content of OSC message
                """
                args, kw = _parse_osc_items(
                    osc_items, 
                    (positional_params, named_params, has_varp, has_varkw), 
                    kwargs
                )

                try:
                    r = maybe_lock(f, lock, address, *args, **kw)
                    # if there was a return value,
                    # send it as a message back to the sender
                    if r is not None:
                        if not hasattr(r, '__len__'):
                            print("""
                            value returned from OSC handler should start with route
                            """)
                        else:
                            client = (
                                client[0] if return_host is None else return_host,
                                client[1] if return_port is None else return_port
                            )
                            print('iipyper OSC return', client, r)
                            self.get_client_by_sender(client).send_message(r[0], r[1:])
                            
                except pydantic.ValidationError as e:
                    print(f'ERROR: iipyper OSC handler:')
                    for info in e.errors(include_url=False):
                        msg = info['msg']
                        loc = info['loc']
                        inp = info['input']
                        print(f'\t{inp.args} {inp.kwargs}')
                        print(f'\t{msg} {loc}')

            self.add_handler(route, handler)
            return f

        return decorator if f is None else decorator(f)
    
    def args(self, route=None, return_host=None, return_port=None):
        """decorate a function as an args-style OSC handler.

        the decorated function should look like:
        def f(route, my_arg, my_arg2, ...):
            ...
        the OSC message will be converted to python types and passed as positional
        arguments.
        """
        # return self._decorate(False, route, return_host, return_port, None)
        return self.handle(route, return_host, return_port, kwargs=False)

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
        if json_keys is not None:
            raise ValueError("""
            OSC.kwargs is deprecated. use OSC.handle and replace json_keys with
            type annotation of function arguments as `object`.
            """)
        return self.handle(route, return_host, return_port)
        # return self._decorate(True, route, return_host, return_port, json_keys)

    def __call__(self, client, *a, **kw):
        """alternate syntax for `send` with client name first"""
        self.send(*a, client=client, **kw)

class Updater():
    '''
    Rate-limited function call
    '''

    def __init__(self, f, count=30):
        self.f = f
        self.count = count
        self.counter = 0

    def __call__(self):
        self.counter += 1
        if self.counter >= self.count:
            self.f()
            self.counter = 0

class ReceiveUpdater:
    '''
    Decouples event handling from updating
    Updating is rate-limited by a counter
    '''

    def __init__(self, f, state=None, count=5, update=False):
        self.f = f
        self.count = count
        self.counter = 0
        self.update = update
        self.state = state

    def set(self, state):
        '''
        Set the Updater's state
        '''
        self.state = state
        self.update = True

    def __call__(self):
        '''
        Update the target function with internal state
        '''
        self.counter += 1
        if not (self.update and
                self.counter > self.count and
                self.state is not None):
            return
        self.f(*self.state)
        self.counter = 0
        self.update = False

class ReceiveListUpdater:
    '''
    Decouples event handling from updating
    Updating is rate-limited by a counter
    Assumes a list[float] instead of *args
    '''

    def __init__(self, f, state=None, count=5, update=False):
        self.f = f
        self.count = count
        self.counter = 0
        self.update = update
        self.state = state

    def set(self, state):
        '''
        Set the Updater's state
        '''
        self.state = state
        self.update = True

    def __call__(self):
        '''
        Update the target function with internal state
        '''
        self.counter += 1
        if not (self.update and
                self.counter > self.count and
                self.state is not None):
            return
        self.f(self.state)
        self.counter = 0
        self.update = False

class OSCReceiveUpdater(ReceiveUpdater):
    '''
    ReceiveUpdater with an OSC handler
    '''

    def __init__(self, osc, address: str, f, state=None, count=10, update=False):
        super().__init__(f, state, count, update)
        self.osc = osc
        self.address = address
        osc.add_handler(self.address, self.receive)

    def receive(self, address, *args):
        # FIXME: ip:port/args
        '''
        v: first argument to the handler is the IP:port of the sender
        v: or you can use dispatcher.map directly
           and not set needs_reply_address=True
        j: can I get ip:port from osc itself?
        v: if you know the sender ahead of time yeah,
           but that lets you respond to different senders dynamically
        '''
        self.set(args[1:])

class OSCReceiveListUpdater(ReceiveListUpdater):
    '''
    ReceiveListUpdater with an OSC handler
    '''

    def __init__(self, osc, address: str, f, state=None, count=10, update=False):
        super().__init__(f, state, count, update)
        self.osc = osc
        self.address = address
        osc.add_handler(self.address, self.receive)

    def receive(self, address, *args):
        self.set(list(args[1:]))


class OSCSend():
    '''
    Non rate-limited OSC send
    '''
    def __init__(self, osc, address: str, f, count=30, client=None):
        self.osc = osc
        self.address = address
        self.f = f
        self.client = client

    def __call__(self, *args):
        self.osc.send(self.address, *self.f(*args), client=self.client)

class OSCSendUpdater():
    '''
    Rate-limited OSC send
    '''

    def __init__(self, osc, address: str, f, count=30, client=None):
        self.osc = osc
        self.address = address
        self.f = f
        self.count = count
        self.counter = 0
        self.client = client

    def __call__(self):
        self.counter += 1
        if self.counter >= self.count:
            self.osc.send(self.address, *self.f(), client=self.client)
            self.counter = 0

class OSCReceiveUpdaters:
    '''
    o = OSCReceiveUpdaters(osc,
        {"/tolvera/particles/pos": s.osc_set_pos,
         "/tolvera/particles/vel": s.osc_set_vel})
    '''

    def __init__(self, osc, receives=None, count=10):
        self.osc = osc
        self.receives = []
        self.count = count
        if receives is not None:
            self.add_dict(receives, count=self.count)

    def add_dict(self, receives, count=None):
        if count is None:
            count = self.count
        {a: self.add(a, f, count=count) for a, f in receives.items()}

    def add(self, address, function, state=None, count=None, update=False):
        if count is None:
            count = self.count
        self.receives.append(
            OSCReceiveUpdater(self.osc, address, function,
                              state, count, update))

    def __call__(self):
        [r() for r in self.receives]


class OSCSendUpdaters:
    '''
    o = OSCSendUpdaters(osc, client="particles", count=10,
        sends={
            "/tolvera/particles/get/pos/all": s.osc_get_pos_all
        })
    '''

    def __init__(self, osc, sends=None, count=10, client=None):
        self.osc = osc
        self.sends = []
        self.count = count
        self.client = client
        if sends is not None:
            self.add_dict(sends, self.count, self.client)

    def add_dict(self, sends, count=None, client=None):
        if count is None:
            count = self.count
        if client is None:
            client = self.client
        {a: self.add(a, f, count=count, client=client)
                     for a, f in sends.items()}

    def add(self, address, function, state=None, count=None, update=False, client=None):
        if count is None:
            count = self.count
        if client is None:
            client = self.client
        self.sends.append(
            OSCSendUpdater(self.osc, address, function, count, client))

    def __call__(self):
        [s() for s in self.sends]


class OSCUpdaters:
    '''
    o = OSCUpdaters(osc, client="boids", count=10,
        receives={
            "/tolvera/boids/pos": b.osc_set_pos,
            "/tolvera/boids/vel": b.osc_set_vel
        },
        sends={
            "/tolvera/boids/pos/all": b.osc_get_all_pos
        }
    )
    '''

    def __init__(self, osc,
                 sends=None, receives=None,
                 send_count=60, receive_count=10,
                 client=None):
        self.osc = osc
        self.client = client
        self.send_count = send_count
        self.receive_count = receive_count
        self.sends = OSCSendUpdaters(
            self.osc, count=self.send_count, client=self.client)
        self.receives = OSCReceiveUpdaters(self.osc, count=self.receive_count)
        if sends is not None:
            self.add_sends(sends)
        if receives is not None:
            self.add_receives(receives)

    def add_sends(self, sends, count=None, client=None):
        if count is None:
            count = self.send_count
        if client is None:
            client = self.client
        self.sends.add_dict(sends, count, client)

    def add_send(self, send, count=None, client=None):
        if count is None:
            count = self.send_count
        if client is None:
            client = self.client
        self.sends.add(send, client=client, count=count)

    def add_receives(self, receives, count=None):
        if count is None:
            count = self.receive_count
        self.receives.add_dict(receives, count=count)

    def add_receive(self, receive, count=None):
        if count is None:
            count = self.receive_count
        self.receives.add(receive, count=count)

    def __call__(self):
        self.sends()
        self.receives()
