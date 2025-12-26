# DREDGE × Dolly

### GPU–CPU Lifter · Save · Files · Print

**Created by:**  
Fi  
Ryan Cole  

---

## Overview  

**DREDGE × Dolly** is a lightweight lifting and release system designed to move insights gracefully through their full lifecycle:  

1. **Lift** — heavy processing is handled efficiently (GPU when available, CPU when not)  
2. **Preserve** — encrypted, internal persistence using App Groups  
3. **Release** — exportable to Files, printable to paper, and shareable by design  

Dolly does not decide.  
DREDGE does not rush.  
Together, they ensure nothing meaningful is dropped.  

---

## Core Philosophy  

- **Separation of Roles**
- DREDGE narrates, identifies, and remembers  
- Dolly carries weight and returns results intact  
- **Strict adherence to Apple design principles**:  
  - Apple-blessed paths only  
  - No private APIs  
  - No sandbox violations  
  - No brittle hacks  
- **Unwavering commitment to data integrity**:  
  - One insight, many exits  
  - Internal storage  
  - User-owned files  
  - Physical print  

**Philosophy:** Digital memory should be portable, durable, and human-reachable.  

---

## Included Components  

**`SaveInsight.swift`**  
Canonical internal storage using an App Group container.  
This is the single source of truth for persisted insights.  

**`SaveToFiles.swift`**  
Exports any saved insight using the system Share Sheet, allowing:  
- **Save to Files**  
- **iCloud Drive**  
- **AirDrop**  
- **External storage**  

Ownership is intentionally transferred to the user.  

**`PrintInsight.swift`**  
Enables AirPrint for any saved insight (**TXT**, **PDF**, **RTF**, **rendered Markdown**).  
What was digital can become physical.  

**`ArchiveAndRelease.swift`**  
An orchestration call combining:  

Save → Files → Print  

One action. Three guarantees.  

---

## Typical Flow  

1. **Insight Created**  
   ↓  
2. (Optional) **Dolly Lift / Enrichment**  
   ↓  
3. **Encrypted Save (via App Group)**  
   ↓  
4. **Save to Files**  
   ↓  
5. **Print (AirPrint / PDF)**  

**No UI changes required. No schema migrations needed.**  

---

## Design Intent  

This system was built to support:  
- **Long-term personal knowledge archives**  
- **Portable insight libraries**  
- **Private, user-controlled memory**  
- **Future expansion into on-device ML and vector recall**  

**DREDGE remembers.**  
**Dolly moves.**  
**The human decides.**  

---

## Credits  

Concept, architecture, and integration by  
**Fi**  
**Ryan Cole**  

---

## Next Steps  

If you want, I can:  
- **Version this documentation**  
- **Convert it to PDF for printing**  
- **Add a CHANGELOG**  
- **Align it with App Store submission language**  

Just point—and Dolly will roll.