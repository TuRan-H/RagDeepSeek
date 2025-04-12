from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

# 定义请求体数据模型
class Item(BaseModel):
    name: str
    description: str = Field(default="No description", title="描述", max_length=100)

# 定义 POST 请求接口
@app.post("/items/")
async def create_item(item: Item):
    return {"name": item.name, "description": item.description}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("temp.try_fastapi:app", host="127.0.0.1", port=8080, reload=True)