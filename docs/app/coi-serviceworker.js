/*! coi-serviceworker — enables `crossOriginIsolated` (and thus SharedArrayBuffer)
    on static hosts like GitHub Pages that can't set COOP/COEP response headers.
    Adapted from github.com/gzuidhof/coi-serviceworker (MIT). Uses COEP
    "credentialless" so cross-origin CDN resources (Pyodide, fonts) still load. */

const coepCredentialless = true;

// Self-hosted Pyodide artifacts (runtime + numpy/scipy/fonttools wheels) are
// immutable per version, so we cache-first them in Cache Storage. The HTTP cache
// can be evicted; this keeps repeat visits instant even across sessions. Bump the
// version suffix to invalidate when PYVER changes.
const PYODIDE_CACHE = "svgym-pyodide-v0.26.2";
const isPyodideAsset = (url) =>
  url.origin === self.location.origin && url.pathname.includes("/pyodide/");

if (typeof window === "undefined") {
  // ---- running as the service worker ----
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil((async () => {
    // drop caches from older Pyodide versions
    const names = await caches.keys();
    await Promise.all(names
      .filter((n) => n.startsWith("svgym-pyodide-") && n !== PYODIDE_CACHE)
      .map((n) => caches.delete(n)));
    await self.clients.claim();
  })()));

  self.addEventListener("fetch", (event) => {
    const r = event.request;
    if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;

    const url = new URL(r.url);

    // Cache-first for the self-hosted Pyodide artifacts.
    if (r.method === "GET" && isPyodideAsset(url)) {
      event.respondWith((async () => {
        const cache = await caches.open(PYODIDE_CACHE);
        const hit = await cache.match(r);
        if (hit) return hit;
        const resp = await fetch(r);
        if (resp && resp.ok && resp.status === 200) cache.put(r, resp.clone());
        return resp;
      })());
      return;
    }

    const request = (coepCredentialless && r.mode === "no-cors")
      ? new Request(r, { credentials: "omit" })
      : r;

    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response.status === 0) return response; // opaque
          const headers = new Headers(response.headers);
          headers.set("Cross-Origin-Embedder-Policy",
            coepCredentialless ? "credentialless" : "require-corp");
          headers.set("Cross-Origin-Opener-Policy", "same-origin");
          return new Response(response.body, {
            status: response.status, statusText: response.statusText, headers,
          });
        })
        .catch((e) => console.error("coi sw fetch:", e))
    );
  });
} else {
  // ---- running as a page <script>: register self as the SW, reload once ----
  (() => {
    if (window.crossOriginIsolated !== false) return;     // already isolated
    if (!window.isSecureContext) {
      console.log("COI: not a secure context; SharedArrayBuffer unavailable");
      return;
    }
    if (!navigator.serviceWorker) return;
    navigator.serviceWorker
      .register(window.document.currentScript.src)
      .then((reg) => {
        reg.addEventListener("updatefound", () => window.location.reload());
        if (reg.active && !navigator.serviceWorker.controller) window.location.reload();
      })
      .catch((err) => console.error("COI register failed:", err));
  })();
}
