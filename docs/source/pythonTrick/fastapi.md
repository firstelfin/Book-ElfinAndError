# FastAPI 框架的使用

## 1、定义app

```python
from fastapi import FastAPI

app = FastAPI(
    title="xxx",
    description="xxx",
    version="1.0.0",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    }
    # docs_url="/documentation"
)
```

## 2、定义输入输出对象

```python
from pydantic import BaseModel


class File(BaseModel):
    fileUrl:  str
    baseUrls: List[Dict]
    imageId:  str
    pointName:          Optional[str] = None
    suppress: Optional[SuppressParam] = None

    def get_demo(self):
        # 可定义一些对请求体处理的内部逻辑
        pass


class ResItem(BaseModel):
    msgReqId: str
```

## 3、设置生命周期操作

```python
import os
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # app init pr
    print("·"*12 + f" {app.title} startup by {os.getpid()} " + "·"*12)
    await warmup_app_init()
    yield
    # app killed post op
    print("·"*12 + f" {os.getpid()} shutdown ok " + "·"*12)
```

这里我们定义了start_up和shutdown的操作，以及使用warm_up做app的启动测试。

## 4、路由函数设置

### 4.1 同步APP

```python
@app.post("/test", response_model=ResItem)
async def infer_item(
    file: File  # 可定义多个，多个参数需要封装为一个对象传参
):
    res_item = ResItem()

    return res_item
```

### 4.2 异步APP

```python
from fastapi import BackgroundTasks

@app.post("/test", response_model=ResItem)
async def infer_item(
    file: File,
    back_task: BackgroundTasks  # 框架会自动给每个请求初始化传参
):
    # deepLearnFunc是需要进行后台处理的对象，param是需要传的参数，这里一般是app入参
    back_task.add(deepLearnFunc, param)
    res_item = ResItem()

    return res_item
```

## 5、获取异步APP的后台任务数量

```python
import asyncio

async def get_task_count():
    loop = asyncio.get_event_loop()
    pending = asyncio.all_tasks(loop=loop)
    return len(pending)
```

欢迎讨论更多的获取方式（GitHub）。

## 6、控制框架等待的请求数量

遗憾是暂时没有调通过，内部直接获取等待处理请求数量的代码。需要在APP层面控制流量，防止客户端发送过多请求导致积压，可以在app启动是设置**limit_concurrency**参数。这个参数在unicorn内是有多处实现的。

## 7、错误代理

### 7.1 请求体验证错误

```python
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def http_exception_handler(request: Request, exec: RequestValidationError):
    return JSONResponse(
        status_code=200,
        content={
            "code": 429000
        }
    )
```

通过使用装饰器向exception_handler注册了请求体错误处理函数。

## 8、启动APP

### 8.1 常规启动

```python
uvicorn.run("xxx:app", host=os.getenv("HOST"), port=int(os.getenv("PORT")), workers=2)
```

### 8.2 Config与Server

```python
from uvicorn import Config, Server

config = Config("async_app:app", host="0.0.0.0", port=9006, workers=2, limit_concurrency=10)
server = Server(config)
server.run()
```


## 9、发布定时任务
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # app init pr
    print("·"*12 + f" {app.title} startup by {os.getpid()} " + "·"*12)
    # 添加定时任务
    scheduler.add_job(
        cache_check,                  # cache是我的缓存处理函数
        "cron", 
        id=9005, 
        name="cacheCheck", 
        replace_existing=True,
        day_of_week="1,3,5,7",        # 每周隔一天执行
        hour=23,                      # 执行时间23:10
        minute=10
    )  # 每天晚上进行检修
    await warmup_app_init()
    yield
    # app killed post op
    scheduler.shutdown()
    print("·"*12 + f" {os.getpid()} shutdown ok " + "·"*12)


async def cache_check() -> None:
    """定时检查缓存文件是否过期"""

    cache_dir, soft_link = await check_cache_env()
    time_stamp = datetime.datetime.now().strftime(r"%y%m%d") - int(os.getenv("CACHE_TIME", 14))
    logger.info("开始执行检修程序...")

    for file in cache_dir.iterdir():
        file_stem_list = file.stem.split("-")
        if int(file_stem_list[-1]) > time_stamp: continue
        soft_link_path = soft_link / f"{file_stem_list[0]}-{file_stem_list[1]}"
        await unlink_file(soft_file_path=soft_link_path)
```

参考 [https://blog.csdn.net/lipachong/article/details/99962134](https://blog.csdn.net/lipachong/article/details/99962134)
