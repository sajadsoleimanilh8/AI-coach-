# Vendored GSAP 3.12.5

`gsap.min.js` and `ScrollTrigger.min.js` from the npm package `gsap@3.12.5`
(`npm pack gsap@3.12.5`, files taken from `package/dist/`). Verified
byte-identical to the cdnjs 3.12.5 builds they replaced.

They were previously two `<script src="https://cdnjs.cloudflare.com/...">` tags
with no integrity attribute, so the landing page could not be built or run
offline and depended on a third party at page load. These are plain classic
scripts served from `public/`, loaded exactly where the CDN tags used to be, so
the global `gsap` / `ScrollTrigger` the inline script expects are unchanged.

To upgrade: `npm pack gsap@<version>`, copy the two files out of
`package/dist/`, and update the version here and in `index.html`'s comment.
