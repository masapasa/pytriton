#%%
import numpy as np
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
def _add_sub(**inputs):
    a_batch, b_batch = inputs.values()
    add_batch = a_batch + b_batch
    sub_batch = a_batch - b_batch
    return {"add": add_batch, "sub": sub_batch}
triton = Triton()
triton.bind(
    model_name="AddSub",
    infer_func=_add_sub,
    inputs=[
        Tensor(name="a", dtype=np.float32, shape=(-1,)),
        Tensor(name="b", dtype=np.float32, shape=(-1,)),
    ],
    outputs=[
        Tensor(name="add", dtype=np.float32, shape=(-1,)),
        Tensor(name="sub", dtype=np.float32, shape=(-1,)),
    ],
    config=ModelConfig(max_batch_size=128),
)
triton.run()
from pytriton.client import ModelClient
batch_size = 2
a_batch = np.ones((batch_size, 1), dtype=np.float32)
b_batch = np.ones((batch_size, 1), dtype=np.float32)
with ModelClient("localhost", "AddSub") as client:
    result_batch = client.infer_batch(a=a_batch, b=b_batch)

for output_name, data_batch in result_batch.items():
    print(f"{output_name}: {data_batch.tolist()}")
# %%
import numpy as np
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
def _add_sub(**inputs):
    a_batch, b_batch = inputs.values()
    add_batch = a_batch + b_batch
    sub_batch = a_batch - b_batch
    return {"add": add_batch, "sub": sub_batch}
triton = Triton()
triton.bind(
        model_name="AddSub",
        infer_func=_add_sub,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="add", dtype=np.float32, shape=(-1,)),
            Tensor(name="sub", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
    )
triton.run()
from pytriton.client import ModelClient
batch_size = 2
a_batch = np.ones((batch_size, 1), dtype=np.float32)
b_batch = np.ones((batch_size, 1), dtype=np.float32)
with ModelClient("localhost", "AddSub") as client:
    result_batch = client.infer_batch(a_batch, b_batch)
for output_name, data_batch in result_batch.items():
    print(f"{output_name}: {data_batch.tolist()}")

# %% [markdown]
# ## Re-setup triton server with modified inference callable

# %% [markdown]
# Stop triton server

# %%
triton.stop()

# %% [markdown]
# Redefine inference callable

# %%
def _add_sub(**inputs):
    a_batch, b_batch = inputs.values()
    add_batch = (a_batch + b_batch) * 2
    sub_batch = (a_batch - b_batch) * 3
    return {"add": add_batch, "sub": sub_batch}

# %% [markdown]
# Load model again

# %%
triton.bind(
        model_name="AddSub",
        infer_func=_add_sub,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="add", dtype=np.float32, shape=(-1,)),
            Tensor(name="sub", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
    )

# %% [markdown]
# Run triton server with new model inference callable

# %%
triton.run()

# %% [markdown]
# ## The same inference performed with modified inference callable

# %%
with ModelClient("localhost", "AddSub") as client:
    result_batch = client.infer_batch(a_batch, b_batch)

for output_name, data_batch in result_batch.items():
    print(f"{output_name}: {data_batch.tolist()}")

# %% [markdown]
# Stop server at the end

# %%
triton.stop()


