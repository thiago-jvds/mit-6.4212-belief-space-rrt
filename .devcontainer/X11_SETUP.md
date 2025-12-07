# X11 Display Forwarding Setup for macOS

This guide explains how to set up X11 forwarding so that matplotlib windows from the Docker container can display on your Mac.

## Prerequisites

1. **Install XQuartz** (X11 server for macOS):
   ```bash
   brew install --cask xquartz
   ```

2. **Configure XQuartz**:
   - Open XQuartz (Applications → Utilities → XQuartz)
   - Go to **XQuartz → Preferences → Security**
   - Check **"Allow connections from network clients"**
   - **Restart XQuartz** (quit completely and reopen - this is important!)

3. **Enable X11 access on your Mac** (REQUIRED):
   
   After XQuartz is running, open a **Terminal on your Mac** (not in the container) and run:
   ```bash
   xhost +localhost
   ```
   
   Or more securely, allow only Docker:
   ```bash
   xhost + 127.0.0.1
   ```
   
   This allows X11 connections from localhost (which Docker Desktop uses).

## Configuration

The devcontainer is already configured with:
- `DISPLAY=host.docker.internal:0` in `devcontainer.json`
- X11 socket volume mount
- X11 libraries installed in `postCreate.sh`

## Testing

After rebuilding the devcontainer:

1. **Make sure XQuartz is running** (check menu bar)

2. **In the container terminal**, test:
   ```bash
   echo $DISPLAY
   # Should show: host.docker.internal:0
   
   # Test with a simple X11 app
   xclock
   # Should open a clock window on your Mac
   ```

## Troubleshooting

### Error: "Authorization required, but no authorization protocol specified"

This means X11 authentication failed. Try these steps:

1. **On your Mac** (not in container), run:
   ```bash
   xhost +localhost
   ```
   Then restart the devcontainer.

2. **Check XQuartz is running**:
   - Look for XQuartz icon in menu bar
   - If not running, start it: Applications → Utilities → XQuartz

3. **Verify XQuartz security settings**:
   - XQuartz → Preferences → Security
   - "Allow connections from network clients" should be checked
   - Restart XQuartz after changing

4. **Try using your Mac's IP directly**:
   - Get your Mac's IP:
     ```bash
     ipconfig getifaddr en0  # or en1 for Wi-Fi
     ```
   - Update `devcontainer.json`:
     ```json
     "runArgs": [
       "-e", "DISPLAY=<YOUR_MAC_IP>:0"
     ]
     ```
   - On your Mac, run: `xhost +<YOUR_MAC_IP>`

### If matplotlib windows still don't appear:

1. **Check DISPLAY variable in container**:
   ```bash
   echo $DISPLAY
   ```

2. **Test with xclock**:
   ```bash
   xclock
   ```
   If xclock works, matplotlib should work too.

3. **Check X11 forwarding**:
   ```bash
   # In container
   xauth list
   # May be empty, but that's OK if xhost is used
   ```

4. **Alternative: Use non-GUI backend**:
   If X11 forwarding doesn't work, matplotlib will automatically fall back to a non-GUI backend and print warnings instead of displaying windows.

## Security Note

`xhost +localhost` allows any local process to connect to X11. For better security, you can:
- Use `xhost +127.0.0.1` instead
- Or set up xauth cookies (more complex)

## Quick Start Checklist

- [ ] XQuartz installed and running
- [ ] XQuartz security setting enabled
- [ ] XQuartz restarted after enabling security
- [ ] `xhost +localhost` run on Mac
- [ ] Devcontainer rebuilt
- [ ] `xclock` test works in container
