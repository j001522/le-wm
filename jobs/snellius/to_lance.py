# h5 -> lance (per-frame JPEG blobs, the format upstream LeWM's pusht config reads; ~8x faster loading
# than the downloaded h5, whose pixels are compressed in 100-frame chunks).
# usage: python to_lance.py SRC.h5 DEST.lance [N_EPISODES]   (absolute paths: relative names resolve via HF)
import sys, time, hdf5plugin  # noqa: F401  (h5 pixel filter)
import stable_worldmodel.data.utils as U
from stable_worldmodel.data.format import get_format
src, dest = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else None
ds = U.load_dataset(src)
N = len(ds.lengths) if n is None else n
t = time.time()
def eps():
    for i in range(N):
        if i % 500 == 0: print(f"{i}/{N} episodes, {time.time()-t:.0f}s", flush=True)
        yield U._episode_to_step_lists(ds.load_episode(i), int(ds.lengths[i]))
with get_format("lance").open_writer(dest) as w:
    w.write_episodes(eps())
print(f"done {N} episodes in {time.time()-t:.0f}s", flush=True)
