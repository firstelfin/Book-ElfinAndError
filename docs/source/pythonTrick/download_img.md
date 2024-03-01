# 图片下载

## 1、fastapi的app内部使用concurrent库并发没有生效

concurrent的并发在IO操作上有明显提速，在fastapi内部去请求其他URL也有提速，但是监管发现还是有串行，最后使用[httpx](https://www.python-httpx.org/async/)进行异步操作。

## 2、httpx异步推荐

httpx是requests的加强版，[为什么要使用它？](https://baijiahao.baidu.com/s?id=1700108659857219798&wfr=spider&for=pc)

### 2.1 避免全局变量版本

图片下载代码块：

```python
async def download_img(img_file, client=None):
    """异步下载图片, requests库在多线程里面并发效果差, httpx有明显增益"""
    try:
        if client is None:
            async with httpx.AsyncClient() as client:
                req = await client.get(
                    url=img_file,
                    timeout=TIMEOUT
                )
        else:
            req = await client.get(
                url=img_file,
                timeout=TIMEOUT
            )   
        return req.content
    except Exception as e:
        return f"Client error for '{img_file}', exec error: {str(e)}"
```

调用代码块：

```python
async def download_base_and_dete(base_images):

    client = httpx.AsyncClient()
    res = [await download_img(addr[0] if type(addr)==list else addr["url"], client=client) for addr in base_images]
    await client.aclose()

    return res
```

---

## 3、并行下载

我们根据[async/await介绍](./async_await.md)得知，上面的优化是有的，我常规的concurrent多线程是并发处理，涉及线程切换，这个操作本质上也是一个耗时操作；而使用异步(协程)处理本质还是并发，只是少了一些切换的操作。即使协程已经非常优秀了，但是实际使用过程中，我们遇到的是高IO的操作，这里推荐使用[multiprocessing](https://docs.python.org/zh-cn/3/library/multiprocessing.html).

[测试参考资源](https://blog.csdn.net/BobYuan888/article/details/109266020)


### 3.1 map并行

```python
from multiprocessing.dummy import Pool as ThreadPool


def httpx_get(url: str):
    """基于httpx库的GET请求

    :param str url: 图片请求url
    :return [btyes, None]: 图片的二进制流
    """
    try:
        req = httpx.get(url=url, timeout=TIMEOUT)
        res = req.content
        return res
    except:
        return None


async def download_base_and_dete2(file: File):
    """检测图与底图下载

    :param File file: 入参对象
    :return list[btyes]: 所有图片字节流
    """
    assert len(file.baseUrls) > 0, f"Client baseUrls error: excepted length > 1, got {file.baseUrls}."
    assert type(file.baseUrls[0]) == dict, f"Client baseUrls typeError: excepted dict, got {type(file.baseUrls[0])}."

    urls = [file.fileUrl] + [burl["url"] for burl in file.baseUrls]
    pool = ThreadPool(POOL_NUM)
    res = pool.map(httpx_get, urls)
    pool.close()
    pool.join()
    # TODO: 检测每一张图是否下载成功

    return res
```

> 经过测试下载耗时减少了60%左右

---

### 3.2 imap并行

```python
def download_imap(urls):
    """检测图与底图下载

    :param File file: 入参对象
    :return list[btyes]: 所有图片字节流
    """

    pool = ThreadPool(POOL_NUM)
    res = pool.imap(httpx_get, urls, 8)
    pool.close()
    pool.join()
    # TODO: 检测每一张图是否下载成功

    return res
```

对于很长的迭代器，给 chunksize 设置一个很大的值会比默认值 1 极大 地加快执行速度。

> 测试中发现：imap与map效果一样，可能是请求数量不够（reqNum:48）

---

### 3.3 map_async

```python
async def download_map_async(urls):
    """检测图与底图下载

    :param File file: 入参对象
    :return list[btyes]: 所有图片字节流
    """

    pool = ThreadPool(POOL_NUM)
    res = pool.map_async(httpx_get, urls)
    pool.close()
    pool.join()
    # TODO: 检测每一张图是否下载成功

    return res


def test2():
    res = download_map_async(LOCAL_URLS)
    res = asyncio.run(res).get()
    return res
```

> 异步测试时间比map、imap要稳定一些，时间就快一秒左右（reqNum:48  * 10次循环）

