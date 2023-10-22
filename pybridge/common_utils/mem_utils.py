"""
@file:mem_utils
@author:qzz
@date:2023/2/20
@encoding:utf-8
"""


def get_mem_usage() -> str:
    """
    Get the memory usage message
    Returns:
        A string describes available, used and free memory.
    """
    import psutil

    mem = psutil.virtual_memory()
    result = ""
    result += "available: %s, " % (mem2str(mem.available))
    result += "used: %s, " % (mem2str(mem.used))
    result += "free: %s" % (mem2str(mem.free))
    # result += "active: %s\t" % (mem2str(mem.active))
    # result += "inactive: %s\t" % (mem2str(mem.inactive))
    # result += "buffers: %s\t" % (mem2str(mem.buffers))
    # result += "cached: %s\t" % (mem2str(mem.cached))
    # result += "shared: %s\t" % (mem2str(mem.shared))
    # result += "slab: %s\t" % (mem2str(mem.slab))
    return result


def mem2str(num_bytes: int) -> str:
    assert num_bytes >= 0
    if num_bytes >= 2 ** 30:  # GB
        val = float(num_bytes) / (2 ** 30)
        result = "%.3f GB" % val
    elif num_bytes >= 2 ** 20:  # MB
        val = float(num_bytes) / (2 ** 20)
        result = "%.3f MB" % val
    elif num_bytes >= 2 ** 10:  # KB
        val = float(num_bytes) / (2 ** 10)
        result = "%.3f KB" % val
    else:
        result = "%d bytes" % num_bytes
    return result
