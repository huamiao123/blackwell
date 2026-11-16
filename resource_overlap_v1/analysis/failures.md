# Failure and protocol-adjustment log

1. `gate1_nvfp4_b8_s1`: first baseline invocation stopped on the new assertion
   that *all scheduler calls* must equal prefill+256. The preceding output-count,
   Decode-batch, Decode-count and 256 Graph-replay assertions passed. No formal
   result retained. Async scheduling can call the scheduler with an empty batch.
   Revision keeps every call, counts **nonempty** calls for shape validation, and
   saves raw state on any later failure. A fresh run is required; the failed run
   is not rescued as a performance sample.

2. Fresh baselines verified the extra call: 258 total scheduler calls, 257
   nonempty calls (one full prefill + 256 Decode), and 256 Graph replays.
   All four successful configurations have the same counts.

3. `gate2_fp32_low`: rejected before Popen/CUDA with
   `AUTO_LAUNCH_CONTROLLER_PRESENT_NO_RUN`. Read-only observation at
   2026-11-17 00:03:33 +08:00 then found another user's PID 527046 using
   23752 MiB, GPU utilization 47%, and none of our experiment processes.
   No probe result exists. This is an external scheduling stop, not a negative
   hardware result. See `blocked_attempt.json` and `stop_status.json`.

4. A CPU deployment overwrote the tail of aggregate `commands.txt` with an older
   local copy. All five separate supervision logs retained full commands and
   metadata, and all four measurement JSON files remained intact. The original
   aggregate is preserved; `commands_reconciled.jsonl` derives the complete five
   workload invocations from those logs. Deployment no longer writes runtime
   root logs. This is an artifact-aggregation error, not a timing-data repair.
