import pytest
import time
from collections import defaultdict

from iipyper import OSC
from iipyper.types import *

@pytest.fixture(scope='module')
def setup_osc():
    port = 9999
    osc = OSC(port=port)
    osc.create_client('self', '127.0.0.1', port)
    return osc

def test_send_rcv(setup_osc):
    osc = setup_osc

    _items = (777, None, 'abc', [1,2,3])
    status = {'rcv':False}

    @osc.handle
    def route_name(route, *items):
        assert route == '/route_name'
        assert items == _items
        status['rcv'] = True

    osc.send('/route_name', *_items)

    time.sleep(0.02)
    assert status['rcv'], 'OSC not received'


def test_parse(setup_osc):
    # postional Splat[None] followed by keyword Splat[None]
    # object-annotated JSON string
    osc = setup_osc

    _x = [1,2,3]
    _y = ['a', 'b', 'c']
    _z = {'key':0, 'key2':[0,1,2], 'key3':{}}
    status = {'rcv':False}

    @osc.handle
    def route_name2(route, x:Splat[None], y:Splat[None], z:object):
        assert route == '/route_name2'
        assert x == _x
        assert y == _y
        assert z == _z
        status['rcv'] = True

    osc.send('/route_name2', *_x, 'y', *_y, 'z', json.dumps(_z))

    time.sleep(0.02)
    assert status['rcv'], 'OSC not received'
    

def test_parse2(setup_osc):
    # VARARGS following Splat[N]
    # default value of JSON arg
    osc = setup_osc

    _x = (1,2,3)
    _y = ['a', 'b', 'c']
    _z = None
    _items = (777, None, 'abc', [1,2,3])
    status = {'rcv':False}

    @osc.handle
    def route_name3(route, x:Splat[3], *items, y:Splat[None]=[], z:object=_z):
        assert route == '/route_name3'
        assert x == _x
        assert items == _items
        assert y == _y
        assert z == _z
        status['rcv'] = True

    osc.send('/route_name3', *_x, *_items, 'y', *_y)

    time.sleep(0.02)
    assert status['rcv'], 'OSC not received'


def test_send_rcv_long(setup_osc):
    osc = setup_osc

    _items = (1,)*1024
    status = {'rcv':False}

    @osc.handle
    def long_long_stupidly_long_route_name(route, *items):
        assert route == '/long_long_stupidly_long_route_name'
        assert items == _items
        status['rcv'] = True

    osc.send('/long_long_stupidly_long_route_name', *_items)

    time.sleep(0.02)
    assert status['rcv'], 'OSC not received'


def test_send_rcv_long2(setup_osc):
    osc = setup_osc

    _items = [1]*1024
    status = {'rcv':False}

    @osc.handle('/long/long/stupidly/long/route/name/its/so/long')
    def _(route, items:Splat[None]):
        assert route == '/long/long/stupidly/long/route/name/its/so/long'
        assert items == _items
        status['rcv'] = True

    osc.send('/long/long/stupidly/long/route/name/its/so/long', *_items)

    time.sleep(0.02)
    assert status['rcv'], 'OSC not received'

@pytest.mark.parametrize('case', [
    ('/a', ('/a',), ('/a/b',)),
    ('/a/*', ('/a/b', '/a/', '/a/b/c/'), ('/a',)),
    ('/*/b', ('/a/b', '//b'), ('/a/c/b')),
    ('/a/*/b', ('/a/c/b', '/a//b'), ('/a/c/d/b')),
    ('/a/**/b', ('/a/c/b', '/a//b', '/a/c/d/b'), ('/a/b')),
    ('/a*/b', ('/a/b', '/ac/b'), ('/a/c/b', '/a/b/c')),
    # ('/a//b', ('/a/c/b', '/a/c/d/b', '/a/b'), ('/ac/b', '/a/b/c', '/a/d/b/c')), # python-OSC does not support //
])
def test_wildcard(setup_osc, case):
    osc = setup_osc

    address, pos, neg = case
    rcv = set()

    @osc.handle(address)
    def _(addr):
        rcv.add(addr)

    time.sleep(0.02)

    for r in pos:
        osc.send(r)
    for r in neg:
        osc.send(r)

    time.sleep(0.02)

    for r in pos:
        assert r in rcv, f'for {address=}, message {r} should have been received'
    for r in rcv:
        assert r in pos, f'for {address=}, message {r} should not have been received'


