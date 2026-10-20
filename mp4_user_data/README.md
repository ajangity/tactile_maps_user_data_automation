# mp4_user_data

Every participant video. Names are `S-<N>_<Referent>.mp4`:

- `S-<N>` is the participant ID (`S-1` is the first participant).
- `<Referent>` is what they were looking for.

Each video uses one map in [`../png_reference_maps`](../png_reference_maps). To find it, look up participant S-N's row for that referent in column B of the Results sheet. See that folder's README for the full rules.

| Video suffix | Results sheet label |
|---|---|
| Corridor | Corridor |
| Door | Door |
| Emergency (S-1 uses `EmergencyExit`) | Emergency Exit |
| Stair | Stair |
| Elevator | Elevator |
| Bathroom | Bathroom |
| Inaccessible | Inaccessible Area |
| Dangerous | Dangerous Area |
| Reception | Reception |
