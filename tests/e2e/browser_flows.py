#!/usr/bin/env python3
"""Headless browser flows for mode UIs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import List, Optional

try:
    from playwright.async_api import async_playwright  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    async_playwright = None


_CAPTURE_SCRIPT = r"""
(() => {
  window.__oe_sent_frames = [];
  const originalSend = WebSocket.prototype.send;
  WebSocket.prototype.send = function(data) {
    try {
      if (typeof data === "string") {
        window.__oe_sent_frames.push(data);
      } else if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
        window.__oe_sent_frames.push("__binary__");
      } else {
        window.__oe_sent_frames.push(String(data));
      }
    } catch (_) {
      window.__oe_sent_frames.push("__capture_error__");
    }
    return originalSend.call(this, data);
  };
})();
"""


@dataclass
class BrowserFlowResult:
    executed: bool
    passed: bool
    actions_sent: List[str]
    warning: Optional[str] = None
    error: Optional[str] = None


def browser_available() -> bool:
    return async_playwright is not None


async def run_browser_flow(mode: str, base_url: str) -> BrowserFlowResult:
    if async_playwright is None:
        return BrowserFlowResult(
            executed=False,
            passed=False,
            actions_sent=[],
            warning=(
                "playwright is not installed. "
                "Install with: pip install playwright && playwright install chromium"
            ),
        )

    expected_actions = _expected_actions(mode)
    sent_actions: List[str] = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            await page.add_init_script(_CAPTURE_SCRIPT)
            await page.goto(base_url, wait_until="domcontentloaded")

            await page.wait_for_selector("#state", timeout=20_000)
            await page.wait_for_function(
                "() => document.querySelector('#state') && "
                "document.querySelector('#state').textContent.includes('Connected')",
                timeout=30_000,
            )

            if mode == "conversation":
                await _run_conversation_flow(page)
            elif mode == "security":
                await _run_security_flow(page)
            elif mode == "beauty":
                await _run_beauty_flow(page)
            else:
                raise ValueError(f"unsupported mode: {mode}")

            await asyncio.sleep(1.0)
            raw_frames = await page.evaluate("() => window.__oe_sent_frames || []")
            sent_actions = _extract_actions(raw_frames)

            await context.close()
            await browser.close()
    except Exception as exc:
        return BrowserFlowResult(
            executed=True,
            passed=False,
            actions_sent=sent_actions,
            error=str(exc),
        )

    missing = [a for a in expected_actions if a not in sent_actions]
    return BrowserFlowResult(
        executed=True,
        passed=not missing,
        actions_sent=sent_actions,
        error=f"missing UI actions: {', '.join(missing)}" if missing else None,
    )


async def _run_conversation_flow(page) -> None:
    await page.fill("#textInput", "e2e conversation ping")
    await page.click("#sendBtn")
    await page.click("#srcScreen")
    await page.click("#srcCamera")
    await page.click("#toggleVision")
    await page.click("#toggleVision")
    await page.dispatch_event("#ptt", "mousedown")
    await page.dispatch_event("#ptt", "mouseup")


async def _run_security_flow(page) -> None:
    await page.click("#toggleSecurity")
    await page.click("#toggleSecurity")
    await page.click("#refreshEvents")
    await page.fill("#analyzeTarget", "auto")
    await page.fill("#prompt", "Summarize anomalies from the last 10 seconds.")
    await page.click("#analyzeBtn")


async def _run_beauty_flow(page) -> None:
    await page.click("#toggleBeauty")
    await page.click("#toggleBeauty")
    await _set_range_value(page, "#smooth", 44)
    await _set_range_value(page, "#tone", 27)
    await _set_range_value(page, "#brightness", 15)
    await _set_range_value(page, "#warmth", -12)
    await page.click("#resetBtn")


async def _set_range_value(page, selector: str, value: int) -> None:
    await page.eval_on_selector(
        selector,
        "(el, v) => { el.value = String(v); "
        "el.dispatchEvent(new Event('input', { bubbles: true })); }",
        value,
    )


def _extract_actions(raw_frames: List[str]) -> List[str]:
    actions: List[str] = []
    for frame in raw_frames:
        if not isinstance(frame, str) or not frame.startswith("{"):
            continue
        try:
            payload = json.loads(frame)
        except json.JSONDecodeError:
            continue
        action = payload.get("action")
        if isinstance(action, str):
            actions.append(action)
    return actions


def _expected_actions(mode: str) -> List[str]:
    if mode == "conversation":
        return [
            "switch_mode",
            "select_conversation_source",
            "toggle_video_conversation",
            "text_input",
            "push_to_talk",
        ]
    if mode == "security":
        return [
            "switch_mode",
            "toggle_security_mode",
            "security_list_recordings",
            "security_vlm_analyze",
        ]
    if mode == "beauty":
        return [
            "switch_mode",
            "toggle_beauty",
            "set_beauty_skin",
            "set_beauty_light",
        ]
    raise ValueError(f"unsupported mode: {mode}")
