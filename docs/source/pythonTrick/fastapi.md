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
from pydantic import BaseModel, model_validator


class File(BaseModel):
    fileUrl:  str
    baseUrls: List[Dict]
    imageId:  str
    pointName:          Optional[str] = None
    suppress: Optional[SuppressParam] = None
    num:                        float = 0.25
    min_num:                    float = 0.
    max_num:                    float = 3.

    def get_demo(self):
        # 可定义一些对请求体处理的内部逻辑
        pass

    @model_validator(mode="after")
    @classmethod
    def valid(cls, values):
        """设置参数范围检查"""
        num = values.num
        a = values.min_num
        b = values.max_num
        if num < a: values.num = a
        if num > b: values.num = b
        return values


class ResItem(BaseModel):
    msgReqId: str
```

自定义对象, 默认是没有参数范围检查的，我们需要使用model_validator装饰器进行检查。`mode="after"`是设置BaseModel生成好类属性后再进行操作。这样可以实现框架在解析过程中自动帮助我们验证参数范围，与fastapi自带的参数解析不一样，我们在valid方法中对超出范围的数值进行了截断操作(如查询参数Query使用ge、gt、le、lt参数设置取之范围，不满足范围会直接解析报错)。

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
    await warmup_app_init()
    yield
    # app killed post op
    scheduler.shutdown()
    print("·"*12 + f" {os.getpid()} shutdown ok " + "·"*12)


def cache_check() -> None:
    """定时检查缓存文件是否过期"""

    cache_dir, soft_link = check_cache_env()
    time_stamp = datetime.datetime.now().strftime(r"%y%m%d") - int(os.getenv("CACHE_TIME", 14))
    logger.info("开始执行检修程序...")

    for file in cache_dir.iterdir():
        file_stem_list = file.stem.split("-")
        if int(file_stem_list[-1]) > time_stamp: continue
        soft_link_path = soft_link / f"{file_stem_list[0]}-{file_stem_list[1]}"
        unlink_file(soft_file_path=soft_link_path)

# 添加定时任务
scheduler.add_job(
    cache_check,                  # cache是我的缓存处理函数，注意如果使用async函数可能会有问题
    "cron", 
    id="9005", 
    name="cacheCheck", 
    replace_existing=True,
    day_of_week="1,3,5",        # 每周隔一天执行, 从0开始
    hour=23,                      # 执行时间23:10
    minute=10
)  # 每天晚上进行检修
    
"""
可以使用魔法方法添加
@scheduler.scheduled_job(
    "cron", 
    id="cache_check", 
    name="cacheCheck",
    day_of_week="1,3,4,5",
    hour=11,
    minute="0-59",
    second=0
)
"""
```


## 10、Form数据与Query

form-data数据类型不能和json共存。遗憾的是全Form类型参数也不能使用BaseModel封装为类，我们只能在路由函数里面一个参数一个参数的写。在应用过程中，工程部分参数不是使用Form传递，而是基于查询参数传递。由于工程可能比较任性，他们会传入一些没有定义的参数，所以工程侧给的案例是使用一个Request对象的参数接收传入的信息，实现动态参数传入，再自己解析需要的。

鉴于前面的请求体，本身已经具有接收非必要参数(请求体未定义),所以猜测这里也是可以实现的，这样就避免了隐变量的注入，通过显示的行参定义，让框架做解析，保证错误的及时处理，这样不用自己再次对参数进行验证。

下面是一个测试案例：

```python
@app.put("/")
async def get_label(
    file:        bytes = File2(),
    reqID:         int = Form(-1),
    reqTime:       int = Form(-1),
    model_conf:  float = Query(0.25, ge=0, le=1 strict=False),
    area_thresh: float = Query(10000, ge=800, le=60000, strict=False),
    window_size: float = Query(60, ge=0, le=10000, strict=False)
):
    img = img_decode(file)
    return {"model_conf": model_conf, "area_thresh": area_thresh, "window_size": window_size, "shape": img.shape}

```

> 注意：这里的area_thresh如果是整型，我们如果在形参上定义为: `area_thresh: int = 10000`, 那query参数在请求是如果是area_thresh=30.5，即小数部分不为0,那么参数解析会报错，area_thresh=30.0是可以解析的。

> 注意Query(0.25, ge=0, le=1 strict=False)中设置了参数必须大于等于0小于等于1，越界会报422的错误。


调用案例：

```python

with open("/home/elfin/Figure_2.png", "rb") as f:
    img_bytes = f.read()
    

def put_test2(url, data):
    res = httpx.put(
        url=url,
        params=QueryParams(
            elfin="china", 
            save="cccc", 
            other="other", 
            window_size=320.5,
            model_conf=0.25,
            area_thresh=10000
        ),
        files={"file": data["file"]},
        data={"name": data["name"], "points": [36, 45, 12]},
        timeout=20
    )
    print(res.json())

put_test2(
    url="xxxx",
    data={
        "file": img_bytes,
        "name": "elfin"
    }
)
```


参考 [https://blog.csdn.net/lipachong/article/details/99962134](https://blog.csdn.net/lipachong/article/details/99962134)
