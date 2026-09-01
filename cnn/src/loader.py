"""
Thread-basierter Batch-Lader.

WARUM NICHT DER STANDARD-DATALOADER MIT WORKERN
-----------------------------------------------
In dieser Umgebung (opencv-python 5.0.0.93, torch 2.13, Python 3.10)
verklemmt sich JEDER fork-basierte DataLoader beim ersten Batch, sobald
num_workers > 0 ist - nachgewiesen sowohl mit dem Ultralytics-Loader als auch
mit diesem Datensatz, und unabhaengig von cv2.setNumThreads(0),
deaktiviertem OpenCL, gesetzten OMP/MKL-Limits und der Startmethode 'spawn'.

Threads forken nicht und sind davon nicht betroffen. Sie helfen hier auch
wirklich: die teuren Stufen der Augmentierung (cv2.resize, cv2.cvtColor,
cv2.LUT, grosse numpy-Kopien) geben die GIL frei, laufen also echt parallel.

Gemessen bei imgsz 1024, Batch 16, Mosaic aktiv, Prefetch 3:
     1 Thread    31.9 ms/Sample  -> 3.19 min/Epoche
     4 Threads    5.0 ms/Sample  -> 0.50 min/Epoche
     8 Threads    3.0 ms/Sample  -> 0.30 min/Epoche
    12 Threads    2.5 ms/Sample  -> 0.25 min/Epoche
    16 Threads    2.3 ms/Sample  -> 0.23 min/Epoche
Damit ist die Datenaufbereitung kein Engpass mehr; die GPU limitiert.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class ThreadedBatchLoader:
    """Liefert Batches wie ein DataLoader, aber mit Threads statt Prozessen.

    Aufbau: ein ThreadPoolExecutor bearbeitet EINZELNE Samples, nicht ganze
    Batches. Dadurch parallelisiert auch die Aufbereitung innerhalb eines
    Batches. Die Reihenfolge bleibt erhalten, weil die Futures in
    Einreihungsreihenfolge abgeholt werden - das Training ist damit bei festem
    Seed reproduzierbar.

    Der Vorlauf ist auf prefetch * batch Samples begrenzt. Bricht der
    Verbraucher vorzeitig ab, wartet der Kontextmanager nur auf diese
    begrenzte Zahl offener Futures.

    ds        Datensatz mit __getitem__ / __len__
    batch     Batchgroesse
    collate   Funktion, die eine Liste von Samples zu einem Batch macht
    threads   Anzahl paralleler Arbeitsthreads
    prefetch  Vorlauf in Batches
    """

    def __init__(self, ds, batch, collate, shuffle=True, threads=8,
                 prefetch=3, drop_last=True, seed=0):
        self.ds, self.batch, self.collate = ds, batch, collate
        self.shuffle, self.threads = shuffle, max(1, threads)
        self.prefetch, self.drop_last = max(1, prefetch), drop_last
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return (len(self.ds) // self.batch if self.drop_last
                else -(-len(self.ds) // self.batch))

    def _order(self):
        idx = np.arange(len(self.ds))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(idx)
        return idx[:len(self) * self.batch]

    def __iter__(self):
        flat = [int(i) for i in self._order()]
        self.epoch += 1
        n_ahead = self.prefetch * self.batch
        with ThreadPoolExecutor(max_workers=self.threads) as ex:
            futs, p = deque(), 0
            while p < len(flat) and len(futs) < n_ahead:
                futs.append(ex.submit(self.ds.__getitem__, flat[p]))
                p += 1
            for _ in range(len(self)):
                samples = []
                for _ in range(self.batch):
                    f = futs.popleft()
                    if p < len(flat):
                        futs.append(ex.submit(self.ds.__getitem__, flat[p]))
                        p += 1
                    samples.append(f.result())
                yield self.collate(samples)


if __name__ == "__main__":
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cv2
    cv2.setNumThreads(0)
    from dataset import PipeDataset, collate

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ds = PipeDataset(os.path.join(root, "synth/train/images"), imgsz=1024,
                     cache=True, seed=0)
    for i in range(200):
        ds._load(i)
    for th in (1, 4, 8, 12, 16):
        dl = ThreadedBatchLoader(ds, 16, collate, shuffle=False, threads=th,
                                 prefetch=3)
        it = iter(dl)
        for _ in range(4):            # Vorlauf fuellen, dann messen
            next(it)
        t, n = time.time(), 20
        for _ in range(n):
            next(it)
        dt = (time.time() - t) / n
        print(f"  {th:2d} Threads: {dt * 1000:6.0f} ms/Batch(16) = "
              f"{dt / 16 * 1000:5.1f} ms/Sample -> "
              f"{dt * 375 / 60:4.2f} min/Epoche")
        del it, dl
