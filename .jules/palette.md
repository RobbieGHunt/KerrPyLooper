## 2026-05-18 - Tooltips for dynamically disabled buttons
**Learning:** Users often get confused and feel stuck when main action buttons are disabled without context.
**Action:** Always add a tooltip explaining exactly what steps the user needs to take to enable a disabled button, rather than just leaving it grayed out.

## 2026-05-25 - Contextual Tooltips for Dynamic UI States
**Learning:** In complex tools like `drift_corrector.py`, dynamically toggling `setToolTip()` alongside `setEnabled()` significantly reduces user friction by explicitly stating prerequisite actions (e.g., "Set a reference image and drag a ROI first") rather than leaving them to guess why an action is blocked.
**Action:** When disabling an action button programmatically based on application state, always pair it with a contextual tooltip explaining the required steps to unblock it. Update the tooltip to a generic action description once enabled.
