local pose_base = 0x00DF
local player_health = 0x04A6
local DEATH_POSE = 0x7F

function turn_on_god_mode()
	for i = 0, 3 do
		local pose_addr = pose_base + i
		emu.write(pose_addr, DEATH_POSE, emu.memType.nesMemory)
		emu.write(player_health, 0x30, emu.memType.nesMemory)
	end
end

emu.addEventCallback(turn_on_god_mode, emu.eventType.endFrame)
