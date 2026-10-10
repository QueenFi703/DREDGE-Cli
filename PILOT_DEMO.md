# Fictional pilot story

`/pilot-demo` is a public, read-only, caption-first walkthrough based on Fi’s ten-scene caseworker narration. The public `/preview` links to it. It is separate from the existing interactive orientation and authenticated casework.

## Operation

- Caption playback starts automatically when the page opens in a visible, JavaScript-enabled browser. No audio plays. Select Pause to inspect the screen.
- Default pacing is approximately 8 minutes 20 seconds, with inspection pauses. Slower and Faster options adjust the elapsed timeline.
- Pause resumes at the same paragraph. Previous, Next, scene selection, changing pace and opening the full narration pause playback. Restart starts the story over automatically.
- Hiding the tab pauses playback. Escape pauses. Leaving the page clears its timer.
- Playback uses a compact viewport layout: the introduction is collapsed, controls and fictional labels remain visible, and captions are split into at most 36-word blocks without changing total timing. Evidence can be scrolled inside its panel for closer inspection.
- The full narration is available below the player.
- Without JavaScript, the opening scene and first caption remain readable. Playback controls stay disabled and the static-view notice explains the limitation.
- This version has no audio, microphone, screen capture or video export. The owner can narrate the captions while recording separately. No personal-voice claim is made.

## Boundaries

Every screen is prepared fictional training material. The January $900 and February $950 records use Jordan Example, matching the existing public fixture. The player makes no API requests or account lookup, reads no live records, saves no case, creates no uploads/OCR/receipts, makes no AI or research call, and does not approve or release anything. Displayed packet status is explicitly illustrative and pending. The original/redacted text pair is a prepared comparison rather than a newly generated copy. The displayed orientation bubble is labeled an illustration and links to the real interactive preview.

The public route does not change authentication, roles, subscriptions, client assignment, paid inference, or independent-review controls. Nothing here is evidence that a production operation succeeded. Existing authorized test runs are unaffected.

## Verification

- DOM: `NODE_PATH=../dredge-repair/node_modules node --test tests/*-dom.test.cjs` in this local environment; normal installed checkout uses `npm run test:studio`.
- Python: `PYTHONPATH=src python -m pytest tests/test_pilot_demo.py tests/test_studio.py tests/test_studio_live.py -q`.
- Inspect desktop and iPhone-sized layouts, all scene controls, screen reader labels and keyboard navigation before release.
- Publication and deployment require separate approval; this implementation is local until that approval is obtained.

## Asset cache versions

The compact player uses `studio_pilot.css?v=compact-2` and `studio_pilot.js?v=compact-2`. Increment both HTML asset revision tags whenever changing player CSS or JavaScript, so an existing browser session cannot combine new markup with cached older player assets. Verify actual loaded styles and caption labels after deployment, not only the Git commit.
