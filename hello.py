import jax

jax.distributed.initialize()
print(jax.device_count())
