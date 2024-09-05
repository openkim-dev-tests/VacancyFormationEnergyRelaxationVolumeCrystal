from collections import OrderedDict
KEY_SOURCE_VALUE = 'source-value'
KEY_SOURCE_UNIT = 'source-unit'
KEY_SOURCE_UNCERT = 'source-std-uncert-value'

def V(value, unit = '', uncert = ''):
    # Generate OrderedDict for JSON Dump
    res = OrderedDict([
        (KEY_SOURCE_VALUE, value),
    ])
    if unit != '':
        res.update(OrderedDict([
            (KEY_SOURCE_UNIT, unit),
        ]))
    if uncert != '':
        res.update(OrderedDict([
            (KEY_SOURCE_UNCERT, uncert)
        ]))
    return res
