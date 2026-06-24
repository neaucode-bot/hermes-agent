const assert = require('node:assert/strict')
const { test } = require('node:test')

const {
  parseLaunchctlGetenvOutput,
  readMacosLaunchctlEnvVar
} = require('./macos-launchctl-env.cjs')

// ── parseLaunchctlGetenvOutput ─────────────────────────────────────────────

test('parseLaunchctlGetenvOutput extracts a trimmed value', () => {
  assert.equal(parseLaunchctlGetenvOutput('/Users/jarvis/hermes/.hermes\n'), '/Users/jarvis/hermes/.hermes')
})

test('parseLaunchctlGetenvOutput returns null for empty output', () => {
  assert.equal(parseLaunchctlGetenvOutput(''), null)
  assert.equal(parseLaunchctlGetenvOutput('\n'), null)
  assert.equal(parseLaunchctlGetenvOutput(null), null)
})

// ── readMacosLaunchctlEnvVar ─────────────────────────────────────────────────

test('readMacosLaunchctlEnvVar returns null off macOS without spawning', () => {
  let spawned = false
  const exec = () => {
    spawned = true
    return ''
  }
  assert.equal(readMacosLaunchctlEnvVar('HERMES_HOME', { platform: 'linux', exec }), null)
  assert.equal(spawned, false)
})

test('readMacosLaunchctlEnvVar queries launchctl getenv and trims the value', () => {
  const calls = []
  const exec = (cmd, args) => {
    calls.push([cmd, args])
    return '/Users/jarvis/hermes/.hermes\n'
  }
  const value = readMacosLaunchctlEnvVar('HERMES_HOME', { platform: 'darwin', exec })
  assert.equal(value, '/Users/jarvis/hermes/.hermes')
  assert.deepEqual(calls, [['launchctl', ['getenv', 'HERMES_HOME']]])
})

test('readMacosLaunchctlEnvVar returns null when launchctl fails', () => {
  const exec = () => {
    throw new Error('launchctl failed')
  }
  assert.equal(readMacosLaunchctlEnvVar('HERMES_HOME', { platform: 'darwin', exec }), null)
})

test('readMacosLaunchctlEnvVar returns null for an empty value', () => {
  const exec = () => '\n'
  assert.equal(readMacosLaunchctlEnvVar('HERMES_HOME', { platform: 'darwin', exec }), null)
})
