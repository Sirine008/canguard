# OBD-II diagnostic workflow (reference notes)

A disciplined order of operations when a vehicle arrives with a lit MIL
(check-engine light):

1. **Read all codes from all modules**, not only the engine ECM: a P0700,
   for example, only says the transmission module has its own code stored.
   Record freeze-frame data (conditions when the code set).
2. **Prioritize**: communication codes (U0xxx) and circuit codes often cause
   downstream performance codes. Fix circuit/communication faults before
   chasing efficiency codes such as P0420 (catalyst) or P0171 (lean).
3. **Verify with live data**: fuel trims, O2 sensor switching, ECT vs IAT at
   cold soak, misfire counters per cylinder, commanded vs actual values.
   Short-term + long-term fuel trim beyond roughly +/-10 % indicates a
   mixture problem worth diagnosing.
4. **Test, don't guess**: swap coil/plug between cylinders to see if a
   misfire follows; smoke-test the intake and EVAP systems for leaks;
   measure fuel pressure under load; scope sensors that only fail hot.
5. **Clear codes and confirm the repair** with a drive cycle, checking that
   the relevant OBD monitors run and complete.

Common pattern groups:

- **P0171/P0174 together** (lean both banks): shared cause — MAF
  under-reporting, large vacuum leak, weak fuel delivery — not two separate
  bank-specific faults.
- **P0420 after misfires**: raw fuel from misfires overheats and degrades
  the catalyst; fix ignition/fuel faults first, then re-evaluate catalyst
  efficiency.
- **Multiple U-codes at once**: suspect the shared bus (wiring, termination,
  a failing node pulling the bus down) rather than several modules failing
  simultaneously; see U0001/U0073 checks.
