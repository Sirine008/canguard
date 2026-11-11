# CAN bus basics (reference notes)

The Controller Area Network (CAN, ISO 11898) is the dominant in-vehicle
network for powertrain and chassis ECUs. Key properties relevant to
diagnostics and security:

- **Broadcast bus**: every node sees every frame; frames carry no source or
  destination address, only an identifier (11-bit standard, 29-bit extended).
- **Arbitration by priority**: when two nodes transmit simultaneously, the
  lower identifier wins arbitration bit-by-bit (a dominant 0 beats a
  recessive 1). An attacker flooding ID 0x000 therefore always wins the bus —
  the basis of CAN denial-of-service attacks.
- **No authentication**: classical CAN has no sender authentication or
  encryption. Any node (or attached attacker) can transmit any identifier,
  which enables spoofing/masquerade attacks on legitimate IDs.
- **Error handling**: nodes that accumulate transmit/receive errors go
  error-passive and eventually **bus-off** (they stop transmitting). DTC
  U0073 reports a control module observing bus-off. A healthy high-speed CAN
  shows ~60 ohm between CAN_H and CAN_L (two 120 ohm terminators in
  parallel).
- **Typical traffic**: ECUs broadcast most frames cyclically (periods of
  5-100 ms). This strong periodicity is what timing-based intrusion
  detection exploits: injected frames violate the learned per-ID period.
- **Physical layer checks**: CAN_H idles ~2.5 V and is driven toward 3.5 V;
  CAN_L idles ~2.5 V and is driven toward 1.5 V. A scope on CAN_H/CAN_L
  quickly reveals shorts to power/ground or a missing terminator.

## OBD-II and diagnostics over CAN

Since 2008 (US), OBD-II runs over CAN (ISO 15765-4) at 500 kbit/s on pins 6
(CAN_H) and 14 (CAN_L) of the SAE J1962 connector. Diagnostic trouble codes
(DTCs) follow SAE J2012 / ISO 15031-6: a letter (P powertrain, C chassis,
B body, U network) followed by four digits; the first digit 0 means a
generic (standardized) code, 1 means manufacturer-specific.
