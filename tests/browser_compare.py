"""Run the paired Studio browser checks against a local research server."""

import argparse
import re
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--chrome")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chrome)
        page = browser.new_page(viewport={"width": 1340, "height": 850})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url.rstrip("/") + "/studio")
        expect(page.get_by_role("heading", name="Reference comparison")).to_be_visible()
        compare = page.get_by_role("button", name="Compare documents")
        expect(compare).to_be_enabled()
        compare.click()
        expect(page.locator("#alignment")).to_have_text("Accepted", timeout=15000)
        expect(page.locator("#score")).to_have_text(re.compile(r"^0\.\d{4}$"))
        expect(page.locator(".heat")).to_be_visible()
        page.get_by_label("Structural difference heatmap").uncheck()
        expect(page.locator(".heat")).to_be_hidden()
        page.get_by_label("Structural difference heatmap").check()
        expect(page.locator(".heat")).to_be_visible()
        page.locator("#opacity").focus()
        page.keyboard.press("Home")
        expect(page.locator("#opacity-value")).to_have_text("0%")
        page.get_by_role("button", name="Zoom in").click()
        expect(page.locator("#zoom-level")).to_have_text("125%")
        page.locator("#reference-view").focus()
        page.keyboard.press("ArrowRight")
        assert page.locator("#reference-view").evaluate("(el) => el.style.getPropertyValue('--x')") == "5%"
        assert page.locator("#candidate-view").evaluate("(el) => el.style.getPropertyValue('--x')") == "5%"
        page.get_by_role("button", name="Before / after").click()
        page.locator("#swipe").focus()
        page.keyboard.press("Home")
        assert page.locator("#candidate-view").evaluate("(el) => el.style.getPropertyValue('--split')") == "0%"
        page.get_by_role("button", name="Side by side").click()

        page.get_by_role("button", name="Different reference").click()
        expect(compare).to_be_enabled()
        compare.click()
        expect(page.get_by_role("heading", name="Visible difference flagged")).to_be_visible(timeout=15000)
        expect(page.locator("#example-note")).to_contain_text("different identities")

        page.get_by_role("button", name="Misaligned").click()
        expect(compare).to_be_enabled()
        compare.click()
        expect(page.get_by_role("heading", name="Inconclusive comparison")).to_be_visible(timeout=15000)
        expect(page.locator("#score")).to_have_text("Unavailable")
        expect(page.locator(".heat")).to_be_hidden()
        expect(page.get_by_role("button", name="Before / after")).to_be_disabled()

        transfer = page.evaluate_handle("""() => {
            const data = new DataTransfer();
            data.items.add(new File(['invalid bytes'], 'malformed.png', {type: 'image/png'}));
            return data;
        }""")
        page.locator('[data-slot="candidate"]').dispatch_event("drop", {"dataTransfer": transfer})
        expect(page.get_by_text("malformed.png")).to_be_visible()
        compare.click()
        expect(page.locator("#comparison-status")).to_contain_text("readable")
        external = Path("web/public/comparison-examples/north-quay.png")
        page.locator('input[name="candidate"]').set_input_files(external)
        expect(compare).to_be_enabled()
        page.set_viewport_size({"width": 360, "height": 780})
        assert page.evaluate("document.documentElement.scrollWidth") <= 360
        page.get_by_role("button", name="Clear", exact=True).click()
        expect(compare).to_be_disabled()
        expect(page.get_by_text("Select a candidate or an example.")).to_be_visible()
        page.get_by_role("link", name="Single-image demo").click()
        expect(page.get_by_role("heading", name="Forensic Studio")).to_be_visible()
        assert not errors, errors
        browser.close()
    print("Reference Studio browser checks passed.")


if __name__ == "__main__":
    main()
