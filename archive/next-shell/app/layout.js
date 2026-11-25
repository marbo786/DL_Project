import Script from "next/script";
import "./shell.css";

export const metadata = {
  title: "ForgeryLens",
  description: "An AI341 research prototype for inspecting document images.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <a className="skip" href="#main">Skip to content</a>
        <header className="top">
          <nav className="wrap nav" aria-label="Main navigation">
            <a className="wordmark" href="/"><span className="mark" aria-hidden="true" />ForgeryLens</a>
            <div className="links">
              <a href="/studio">Studio</a><a href="/research">Research</a><a href="/methodology">Methodology</a>
            </div>
          </nav>
        </header>
        {children}
        <footer className="wrap footer">
          <span>ForgeryLens / AI341 research prototype</span>
          <div><a href="/about">About</a><a href="/privacy">Privacy</a></div>
        </footer>
        <noscript><p className="wrap">Enable JavaScript to use the document viewer.</p></noscript>
        <Script src="/assets/studio.js" type="module" strategy="afterInteractive" />
      </body>
    </html>
  );
}