"""Importable model stand-ins for real spawned-process tests."""
import time
import threading


def chunks(config, model, device, text, generation, results, next_step, stop):
    while not stop.is_set():
        with text.get_lock():
            version = generation.value
            prompt = text.value.decode()
        start = next_step.value
        values = [{"axis.pos": float(i), "generation": version, "prompt": prompt}
                  for i in range(start, start + 20)]
        try:
            results.put(("chunk", version, start, values), timeout=.1)
        except Exception:
            pass
        stop.wait(.01)


def blocked(config, model, device, text, generation, results, next_step, stop):
    threading.Event().wait(60)
