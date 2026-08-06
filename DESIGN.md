---
name: Seestar Superimpose
description: A restrained, system-aware astronomy image processing workbench.
colors:
  primary: "#6269a8"
  primary-hover: "#565d99"
  primary-pressed: "#4b5187"
  primary-soft: "#e9eaf6"
  primary-dark: "#6870b0"
  primary-dark-text: "#a2a8e0"
  window-light: "#f2f3f5"
  surface-light: "#ffffff"
  surface-subtle-light: "#f7f8fa"
  text-light: "#20232a"
  text-muted-light: "#59606c"
  border-light: "#d7dbe2"
  window-dark: "#191b20"
  surface-dark: "#20232a"
  surface-subtle-dark: "#252830"
  text-dark: "#f1f3f7"
  text-muted-dark: "#b9bec8"
  border-dark: "#3a3f49"
  success-light: "#2f7c4a"
  warning-light: "#8a5a12"
  error-light: "#a43f44"
  info-light: "#3e718c"
  success-dark: "#67b17f"
  warning-dark: "#d0a052"
  error-dark: "#df7376"
  info-dark: "#78a9c1"
typography:
  headline:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "20px"
    fontWeight: 650
    lineHeight: 1.25
  title:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "15px"
    fontWeight: 650
    lineHeight: 1.35
  body:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "system default"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "system default"
    fontWeight: 500
    lineHeight: 1.35
rounded:
  control-windows: "5px"
  control-macos: "6px"
  group: "7px"
  panel: "8px"
  drop-zone: "10px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  xxl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#ffffff"
    rounded: "{rounded.control-macos}"
    height: "30px"
    padding: "0 12px"
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
    textColor: "#ffffff"
    rounded: "{rounded.control-macos}"
  button-secondary:
    backgroundColor: "{colors.surface-subtle-light}"
    textColor: "{colors.text-light}"
    rounded: "{rounded.control-macos}"
    height: "30px"
    padding: "0 12px"
  input:
    backgroundColor: "{colors.surface-subtle-light}"
    textColor: "{colors.text-light}"
    rounded: "{rounded.control-macos}"
    height: "30px"
    padding: "0 8px"
  panel:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.text-light}"
    rounded: "{rounded.panel}"
    padding: "16px"
---

# Design System: Seestar Superimpose

## Overview

**Creative North Star: "The Darkroom Workbench"**

The interface behaves like a carefully organized imaging workbench used during a long processing session: calm surfaces, exact labels, visible stage evidence, and a single unmistakable primary action. It borrows the hierarchy and tool confidence of professional photo editors without reproducing their panel density.

The system is restrained rather than monochrome. A low-saturation blue-violet is reserved for primary action, selection, focus, and the current stage. Semantic success, warning, error, and information colors carry real pipeline meaning. Layout uses open alignment and tonal panes; containers exist only where they clarify a functional boundary.

**Key Characteristics:**

- System-native typography and desktop interaction
- Restrained accent usage on less than ten percent of a screen
- Tonal work surfaces with one-pixel structural borders
- Dense where parameters demand it, spacious around decisions
- Shared component language with explicit macOS and Windows metrics

## Colors

The palette uses neutral work surfaces and one desaturated blue-violet brand voice, with matched semantic roles in both system appearances.

### Primary

- **Instrument Violet** (`primary` / `primary-dark`): primary buttons, selected controls, and keyboard focus only. Dark-mode accent text uses `primary-dark-text` so compact labels remain readable.
- **Instrument Violet Soft** (`primary-soft`): selected and running-state backgrounds; never a decorative section fill.

### Neutral

- **Workbench Light** (`window-light`, `surface-light`, `surface-subtle-light`): light-mode application background and pane hierarchy.
- **Workbench Dark** (`window-dark`, `surface-dark`, `surface-subtle-dark`): dark-mode application background and pane hierarchy without pure black chrome.
- **Ink / Silver Ink** (`text-light`, `text-dark`): primary labels and values.
- **Muted Ink** (`text-muted-light`, `text-muted-dark`): secondary descriptions that still meet readable contrast.
- **Structural Line** (`border-light`, `border-dark`): one-pixel divisions, group boundaries, and control outlines.

### Secondary

- **Accepted** (`success-light`, `success-dark`): completed, trustworthy states.
- **Review Amber** (`warning-light`, `warning-dark`): degraded and review-required states.
- **Failure Red** (`error-light`, `error-dark`): destructive action and failure only.
- **Bypass Blue** (`info-light`, `info-dark`): safe passthrough and neutral informational states.

### Named Rules

**The One Instrument Rule.** Instrument Violet appears only on primary action, selection, focus, and active processing state. If a screen looks purple, the rule has been broken.

**The Evidence Rule.** Every semantic color is paired with a symbol and explicit state text; color alone never carries pipeline meaning.

## Typography

**Display Font:** Host system UI font

**Body Font:** Host system UI font

**Label/Mono Font:** Menlo on macOS, Cascadia Mono or Consolas on Windows, monospace fallback for logs only

**Character:** One familiar system family carries the entire product hierarchy. Weight, spacing, and alignment create order; there is no decorative display face.

### Hierarchy

- **Headline** (650, 20px, 1.25): page purpose such as Task Settings.
- **Title** (650, 15px, 1.35): pane and functional-section headings.
- **Body** (400, system default, 1.45): descriptions, state explanations, and form content.
- **Label** (500, system default, 1.35): field labels, buttons, status labels, and compact metadata.
- **Log** (400, 12px): no-wrap diagnostic output only.

### Named Rules

**The Native Voice Rule.** Never bundle or simulate a product font for the desktop UI. macOS and Windows use their own system application font through Qt.

## Elevation

The interface is flat by default. Depth comes from three neutral surface levels, one-pixel borders, and the native window system. No decorative shadows are used; dialogs and menus rely on operating-system elevation.

### Named Rules

**The Structural Surface Rule.** A border or surface change must describe a real pane, input, state banner, or drop target. Content does not receive a container merely to look designed.

## Components

### Buttons

- **Shape:** compact desktop controls with a 6px macOS radius or 5px Windows radius.
- **Primary:** solid Instrument Violet, white label, medium-strong weight; one primary action per context.
- **Secondary:** neutral raised surface and structural border.
- **Destructive:** soft error surface and error-colored label/border; used for Stop, not for ordinary cancellation.
- **Hover / Focus:** tonal hover, pressed fill, and an explicit focus border; disabled state remains legible.

### Chips

- **Style:** stage rows are compact list items, not floating pills. Waiting and skipped remain neutral; running, accepted, passthrough, degraded, failed, and stopped use semantic background plus text.
- **State:** every row includes a symbol, Stage number, short title, state text, and optional detail.

### Cards / Containers

- **Corner Style:** gently bounded functional panes (8px), not a grid of promotional cards.
- **Background:** content, preview, sidebar, inspector, log, and parameter sheet use explicit tonal roles.
- **Shadow Strategy:** no application shadows.
- **Border:** one-pixel structural line.
- **Internal Padding:** 12–16px for panes; 7–10px for compact status rows.

### Inputs / Fields

- **Style:** system-familiar controls on the subtle surface, with a one-pixel border and platform control height (30px macOS, 32px Windows).
- **Focus:** two-pixel Instrument Violet focus border without glow.
- **Error / Disabled:** disabled values stay readable; invalid optional paths use warning copy and preserve safe fallback behavior.

### Navigation

The application uses a stable toolbar, native menu bar, explicit Empty / Task / Run workspace states, and standard platform shortcuts. macOS uses the native application menu and unified toolbar; Windows keeps an in-window menu and Windows control metrics.

### State Banner

Terminal success, review-required, stopped, preparation failure, and processing failure states appear above the run workspace with semantic border/background, explicit copy, and only the relevant report or log action.

## Do's and Don'ts

### Do:

- **Do** use the shared semantic tokens and platform profile instead of adding styles to `main_window.py`.
- **Do** keep Instrument Violet below ten percent of the visible interface.
- **Do** pair every status color with a symbol and state word.
- **Do** use system fonts, standard menus, familiar desktop controls, and Qt focus behavior.
- **Do** let parameter density live in one scrollable full-width sheet while keeping task decisions visible.
- **Do** keep animation limited to real state changes; the current implementation uses no decorative animation.

### Don't:

- **Don't** create card nesting, a full screen of identical cards, or a container for every paragraph.
- **Don't** use decorative glassmorphism, gradient text, neon glow borders, cheap star fields, or blue-purple gradient templates.
- **Don't** add star-field decoration or excessive violet to manufacture “astronomy technology” styling.
- **Don't** build a SaaS hero, hero metrics, landing-page navigation, or web-style responsive typography.
- **Don't** use a colored side-stripe border on alerts or list items.
- **Don't** use excessive corner radius; 10px is the ceiling and is reserved for the drop target.
- **Don't** invent custom scrollbars, switches, menus, or form-control behavior when the platform control is familiar.
- **Don't** copy Lightroom or Affinity Photo's panel complexity; preserve their clarity, not their density.
