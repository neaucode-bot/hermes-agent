// macos-launchctl-env.cjs
//
// Read a user-domain environment variable via `launchctl getenv`.
//
// A GUI app launched from Finder/Dock inherits the environment block captured
// at login, so a variable set via `launchctl setenv` (e.g. in ai.hermes.env.plist)
// AFTER login is invisible in process.env even though a fresh shell — and the
// Hermes CLI — sees it immediately. The desktop's HERMES_HOME resolution relies
// on process.env, so that stale-snapshot gap silently sends the backend to the
// default ~/.hermes. Reading the live launchd user-domain value closes the gap.

const { execFileSync } = require('node:child_process')

// `launchctl getenv <name>` prints the value on stdout (with a trailing newline)
// when set, or nothing when absent. Returns the trimmed value, or null when empty.
function parseLaunchctlGetenvOutput(stdout) {
  if (stdout == null) return null
  const value = String(stdout).trim()
  return value || null
}

// Read a user-domain env var via launchctl. macOS-only: returns null off-darwin
// (without spawning), on any spawn error, or when the value is empty.
function readMacosLaunchctlEnvVar(
  name,
  { platform = process.platform, exec = execFileSync } = {}
) {
  if (platform !== 'darwin' || !name) return null
  let stdout
  try {
    stdout = exec('launchctl', ['getenv', name], {
      encoding: 'utf8',
      timeout: 5000
    })
  } catch {
    return null
  }
  return parseLaunchctlGetenvOutput(stdout)
}

module.exports = {
  parseLaunchctlGetenvOutput,
  readMacosLaunchctlEnvVar
}
