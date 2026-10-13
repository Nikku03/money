import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

from brain import net
from tests.util import TempHome

ROOT = Path(__file__).resolve().parent.parent
# One process asking for SEC turns: prints the time of each turn it gets.
TURNS = """
import sys, time
from brain import net
hosts = sys.argv[1].split(",")
for i in range(int(sys.argv[2])):
    net.wait_turn(hosts[i % len(hosts)], float(sys.argv[3]))
    print(repr(time.time()), flush=True)
"""


class BucketTest(TempHome):
    def test_every_sec_host_is_one_bucket(self):
        for h in ("sec.gov", "www.sec.gov", "data.sec.gov", "efts.sec.gov"):
            self.assertEqual(net.bucket(h), "sec.gov")
        for h in ("query1.finance.yahoo.com", "notsec.gov", "sec.gov.example.com"):
            self.assertEqual(net.bucket(h), h)

    def test_hosts_of_one_bucket_share_the_pace(self):
        sleeps = []
        with mock.patch.object(net.time, "sleep", sleeps.append), mock.patch.dict(net._last, clear=True):
            net.wait_turn("data.sec.gov", 0.15)
            net.wait_turn("www.sec.gov", 0.15)  # another SEC host straight after: it waits
        self.assertEqual(len(sleeps), 1)
        self.assertGreater(sleeps[0], 0.1)
        self.assertTrue((Path(os.environ["MONEY_HOME"]) / "rate.sec.gov").exists())

    def test_processes_take_turns(self):
        gap, n = 0.05, 8
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        procs = [subprocess.Popen([sys.executable, "-c", TURNS, hosts, str(n), str(gap)], cwd=str(ROOT), env=env,
                                  stdout=subprocess.PIPE, text=True)
                 for hosts in ("data.sec.gov", "www.sec.gov,data.sec.gov")]
        times = sorted(float(t) for p in procs for t in p.communicate(timeout=60)[0].split())
        self.assertEqual(len(times), 2 * n)
        gaps = [b - a for a, b in zip(times, times[1:])]
        self.assertGreaterEqual(min(gaps), gap - 0.01, gaps)  # never two turns closer than the gap

    def test_other_hosts_stay_per_process(self):
        start = time.time()
        with mock.patch.dict(net._last, clear=True):
            net.wait_turn("example.test", 0.0)
        self.assertLess(time.time() - start, 0.5)
        self.assertFalse((Path(os.environ["MONEY_HOME"]) / "rate.example.test").exists())


if __name__ == "__main__":
    import unittest
    unittest.main()
