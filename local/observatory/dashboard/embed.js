/*
 * embed.js — additive glue for embedding this dashboard in the Dumè GPT
 * sidebar. Does not modify anything in script.js; reads two query params:
 *
 *   ?embed=1          compact view: hides the dashboard's own sidebar and
 *                      tab bar, shows the (already-default-active) OppChoVec
 *                      map full-bleed, adds an "Open full dashboard" link.
 *   ?commune=<name>    preselects a commune once data has finished loading,
 *                      by calling the dashboard's own afficherCommune() —
 *                      its normal, single entry point for commune selection
 *                      (script.js ~line 2174).
 *
 * Loaded last (after script.js) so afficherCommune already exists; the
 * preselect still polls/retries because the dashboard's own data fetch
 * (chargerFichiersAutomatiquement, on DOMContentLoaded) is async.
 */
(function () {
  "use strict";
  var params = new URLSearchParams(location.search);

  if (params.get("embed") === "1") {
    document.body.classList.add("embed-mode");
    var link = document.createElement("a");
    link.href = "/dashboard/";
    link.target = "_blank";
    link.rel = "noopener";
    link.className = "embed-open-full";
    link.textContent = "Open full dashboard ↗";
    document.body.appendChild(link);
  }

  var commune = params.get("commune");
  if (commune) {
    var attempts = 0;
    (function tryPreselect() {
      attempts++;
      if (typeof window.afficherCommune === "function") {
        try {
          window.afficherCommune(commune);
          var sel = document.getElementById("communeSelect");
          if (sel) {
            for (var i = 0; i < sel.options.length; i++) {
              if (sel.options[i].value === commune) { sel.value = commune; break; }
            }
          }
          return; // done — whether or not the commune was found in the data
        } catch (e) {
          // data likely not loaded yet — fall through to retry
        }
      }
      if (attempts < 25) setTimeout(tryPreselect, 250); // ~6s of retries
    })();
  }
})();
