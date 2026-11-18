# Time-series checks

Along each shot, for the boundary and every scalar above, detect

* **jumps:** frame-to-frame changes exceeding a threshold (e.g. LCFS moves of 50 mm or more) that
  EFIT++ does not show;
* **lag:** the cross-correlation lag between model and EFIT++ series, which exposes the temporal
  window of M9;
* **drift:** a slowly growing offset, found by fitting a low-order trend to the difference;
* **flat-lining or saturation:** stretches where a read-out sits at a clip limit.
