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

