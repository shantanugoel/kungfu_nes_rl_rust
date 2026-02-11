local pose_base = 0x00DF

function kill_all()
	for i = 0, 3 do
		local pose_addr = pose_base + i
		emu.write(pose_addr, DEATH_POSE, emu.memType.nesMemory)
	end
end

emu.addEventCallback(kill_all, emu.eventType.endFrame)
