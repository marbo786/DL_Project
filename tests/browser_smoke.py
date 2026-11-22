"""Exercise the Studio against a running local or hosted web project."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--chrome")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chrome)
        page = browser.new_page(viewport={"width": 1360, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.get_by_role("link", name="Open Forensic Studio").click()
        expect(page.get_by_role("heading", name="Forensic Studio")).to_be_visible()
        page.get_by_role("button", name="Edited specimen").click()
        region = page.locator("#regions .region")
        expect(region).to_be_visible()
        sheet = page.locator("#sheet")
        initial = sheet.bounding_box()
        page.get_by_role("button", name="Zoom in").click()
        zoomed = sheet.bounding_box()
        assert zoomed["width"] > initial["width"]
        box = region.bounding_box()
        metadata = Path("web/public/specimen.json")
        expected_x = json.loads(metadata.read_text())["edit_region"]["x"] if metadata.exists() else 0.09
        assert abs((box["x"] - zoomed["x"]) / zoomed["width"] - expected_x) < 0.01
        page.locator("#viewport").focus()
        page.keyboard.press("ArrowRight")
        assert sheet.bounding_box()["x"] > zoomed["x"]
        page.get_by_role("checkbox", name="Show regions").uncheck()
        expect(region).to_be_hidden()
        page.locator("#file").set_input_files("web/public/original.png")
        expect(page.locator("#notice")).to_contain_text("original.png")
        page.locator("#file").set_input_files(
            {"name": "bad.png", "mimeType": "image/png", "buffer": b"bad"}
        )
        expect(page.locator("#notice")).to_contain_text("could not be opened")
        page.set_viewport_size({"width": 360, "height": 780})
        assert page.evaluate("document.documentElement.scrollWidth") <= 360
        page.get_by_role("link", name="Research", exact=True).click()
        expect(page.get_by_role("heading", name="Research explorer")).to_be_visible()
        assert not errors, errors
        browser.close()
    print("Chromium Studio interaction checks passed.")


if __name__ == "__main__":
    main()