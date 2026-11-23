"""Run the paired Studio browser checks against a local research server."""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def contrast(foreground, background):
    def luminance(hex_color):
        values = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in values]
        return sum(weight * value for weight, value in zip((.2126, .7152, .0722), linear))
    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (light + .05) / (dark + .05)


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
        held = []
        page.route("**/api/v1/compare", lambda route: held.append(route))
        compare.click()
        expect(page.locator("#comparison-status")).to_have_text("Aligning and comparing images.")
        expect(compare).to_be_disabled()
        assert len(held) == 1
        held[0].continue_()
        page.unroute("**/api/v1/compare")
        expect(page.locator("#alignment")).to_have_text("Accepted", timeout=15000)
        expect(page.locator("#score")).to_have_text("0.1062")
        expect(page.locator("#threshold")).to_have_text("0.5850")
        expect(page.locator("#comparison-decision")).to_have_text("No document-level difference flagged")
        expect(page.locator("#evidence-detail")).to_contain_text("Regions can be visible")
        expect(page.locator(".regions")).to_be_visible()
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
        box = page.locator("#candidate-view").bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x - 25, y - 20, steps=4)
        page.mouse.up()
        assert page.locator("#reference-view").evaluate("(el) => el.style.getPropertyValue('--y')") != "0%"
        assert page.locator("#reference-view").evaluate("(el) => el.style.getPropertyValue('--y')") == page.locator("#candidate-view").evaluate("(el) => el.style.getPropertyValue('--y')")
        page.get_by_role("button", name="Before / after").click()
        page.locator("#swipe").focus()
        page.keyboard.press("Home")
        assert page.locator("#candidate-view").evaluate("(el) => el.style.getPropertyValue('--split')") == "0%"
        page.get_by_role("button", name="Side by side").click()

        page.get_by_role("button", name="Different reference").click()
        expect(compare).to_be_enabled()
        compare.click()
        expect(page.locator("#comparison-decision")).to_have_text("Document-level difference flagged", timeout=15000)
        expect(page.locator("#example-note")).to_contain_text("different identities")
        expect(page.locator("#example-note")).to_contain_text("cannot establish reference identity")

        page.get_by_role("button", name="Misaligned").click()
        expect(compare).to_be_enabled()
        compare.click()
        expect(page.locator("#comparison-decision")).to_have_text("Inconclusive comparison", timeout=15000)
        expect(page.locator("#score")).to_have_text("Unavailable")
        expect(page.locator(".heat")).to_be_hidden()
        expect(page.locator("#visual-evidence")).to_be_hidden()
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
        for width, height in ((1340, 850), (820, 1180), (360, 780)):
            page.set_viewport_size({"width": width, "height": height})
            assert page.evaluate("document.documentElement.scrollWidth") <= width
            expect(compare).to_be_visible()
            expect(page.get_by_role("region", name="Document inspection")).to_be_visible()
        tokens = page.evaluate("""() => Object.fromEntries(
            ["--ink", "--muted", "--warning", "--accent", "--base", "--panel", "--viewport"].map(
                key => [key, getComputedStyle(document.documentElement).getPropertyValue(key).trim()]))""")
        for text, background in (("--ink", "--panel"), ("--muted", "--panel"), ("--warning", "--panel"),
                                 ("--accent", "--base"), ("--muted", "--viewport")):
            assert contrast(tokens[text], tokens[background]) >= 4.5
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
