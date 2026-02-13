local pose_base = 0x00DF
local player_health_addr = 0x04A6
local timer_base = 0x0390
local DEATH_POSE = 0x7F

function turn_on_god_mode()
	for i = 0, 3 do
		local pose_addr = pose_base + i
		local timer_addr = timer_base + i
		emu.write(pose_addr, DEATH_POSE, emu.memType.nesMemory)
		emu.write(timer_addr, 0x09, emu.memType.nesMemory)
	end
	emu.write(player_health_addr, 0x30, emu.memType.nesMemory)
end

emu.addEventCallback(turn_on_god_mode, emu.eventType.endFrame)
