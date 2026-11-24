import pytest

from .service_helpers import PASSWORD, add_user
from .test_local_base_models import standalone_server  # noqa: F401
from .test_release_model_flows import _mixed_crop_dataset, _snapshot_rows


def test_browser_confirms_only_the_listed_images_are_excluded(
    standalone_server, client, settings, tmp_path
):
    browser_api = pytest.importorskip("playwright.sync_api")
    from .test_backlog_service_ui import _browser, _login

    ds = _mixed_crop_dataset(client, settings)
    add_user(settings, "alice", PASSWORD)
    problems = []
    with browser_api.sync_playwright() as pw:
        browser, page = _browser(pw, problems)
        try:
            page.on("console", lambda msg: problems.append(msg.text) if msg.type == "error" else None)
            page.on("requestfailed", lambda req: problems.append(req.failure))
            _login(page, standalone_server)
            panel = page.locator("#crop-prep")
            browser_api.expect(panel).to_be_visible()
            for name in ("raw1.png", "raw2.png"):
                browser_api.expect(panel).to_contain_text(name)
            page.locator("#crop-prep-exclude").click()
            confirm = page.get_by_role("button", name="Create the snapshot without them")
            browser_api.expect(confirm).to_be_visible()
            page.screenshot(path=str(tmp_path / "confirm-crop-exclusion.png"))
            with page.expect_response(lambda r: r.request.method == "POST"
                                      and r.url.endswith(f"/datasets/{ds}/snapshots")) as event:
                confirm.click()
            assert event.value.status == 201, event.value.text()
            assert set(_snapshot_rows(settings, event.value.json()["id"])) == {"boxed.png", "studio.png"}
            assert set(event.value.request.post_data_json["exclude_images"]) == {"raw1.png", "raw2.png"}
            assert not problems, problems
        finally:
            browser.close()
