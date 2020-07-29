module SeamCarving

using ImageFiltering
using Images
using ProgressMeter
using ColorSchemes

export SeamCarver, getEnergyImage, calculateEnergyMap, carve_horizontal, paint_horizontal

energyDict = Dict(
    :SobelX => parent(Kernel.sobel()[1]),
    :SobelY => parent(Kernel.sobel()[2]),
    :Sobel  => parent(Kernel.sobel()[1]) + parent(Kernel.sobel()[2]),
    :Scharr => parent(Kernel.scharr()[1]) + parent(Kernel.scharr()[2])
)

function getRow(energyMap :: Array{T, 2}, position :: Int64) where T; end
function getColumn(energyMap :: Array{T, 2}, position :: Int64) where T; end

# :horizontal → reduce the X size
# :vertical   → reduce the Y size
stepDict = Dict(
    :horizontal => (1, 2, getRow, vcat),
    :vertical   => (2, 1, getColumn, hcat)
)

mutable struct SeamCarver
    image :: Array{RGB{T}, 2} where T
    energyKernel :: Array{Float64, 2}
    energyMapDict :: Dict{Symbol, Array{Float64, 2}}
end

function SeamCarver(image :: Array{RGB{T}, 2}, energy :: Symbol) where T
    @assert haskey(energyDict, energy)
    temp_carver = SeamCarver(image, energyDict[energy], Dict(:empty => [1 1]))
    SeamCarver(image, energyDict[energy],
               Dict(:horizontal => calculateEnergyMap(temp_carver, :horizontal),
                    :vertical   => calculateEnergyMap(temp_carver, :vertical)))
end

function getEnergyImage(seamCarver :: SeamCarver; gray :: Bool = true)
    filteredImage = ImageFiltering.imfilter(seamCarver.image,
                                            seamCarver.energyKernel)
    if gray
        return Gray.(filteredImage)
    end
    return filteredImage
end

function get(energyMap :: Array{Float64, 2}, orientation :: Symbol,
             position :: Tuple{Int64, Int64})
    @assert haskey(stepDict, orientation)
    step = stepDict[orientation]
    return energyMap[position[step[1]], position[step[2]]]
end

function increment!(energyMap :: Array{Float64, 2}, orientation :: Symbol,
              position :: Tuple{Int64, Int64}, newVal :: Float64)
    @assert haskey(stepDict, orientation)
    step = stepDict[orientation]
    energyMap[position[step[1]],position[step[2]]] += newVal
end

function calculateEnergyMap(seamCarver :: SeamCarver,
                            orientation :: Symbol) where T
    @assert haskey(stepDict, orientation)
    step = stepDict[orientation]

    energyMap = Float64.(getEnergyImage(seamCarver; gray = true))

    mainAxis          = step[1]
    mainAxisSize      = size(energyMap)[mainAxis]
    secondaryAxis     = step[2]
    secondaryAxisSize = size(energyMap)[secondaryAxis]

    for i ∈ 2:mainAxisSize
        Threads.@threads for j ∈ 1:secondaryAxisSize
            possibilities = (
                [i-1 j] - [0 1],
                [i-1 j],
                [i-1 j] + [0 1],
            )

            if j == 1; possibilities = possibilities[2:3] end
            if j == secondaryAxisSize; possibilities = possibilities[1:2] end

            possibleValues = [get(energyMap, orientation, Tuple(poss))
                              for poss ∈ possibilities]
           
            toSum = minimum(possibleValues)
            increment!(energyMap, orientation, (i, j), toSum)
        end
    end
    return energyMap/minimum(energyMap)
end

function getRow(energyMap :: Array{T, 2}, position :: Int64) where T
    return energyMap[position, :]
end

function getColumn(energyMap :: Array{T, 2}, position :: Int64) where T
    return energyMap[:, position]
end

function my_hcat(arr1, arr2)
    if isempty(arr1)
        return arr2
    end
    if isempty(arr2)
        return arr1
    end
    return hcat(arr1', arr2')
end

function carve_horizontal(seamCarver :: SeamCarver;
               numCarves :: Int64 = 1)

    energy = seamCarver.energyMapDict[:horizontal]
    @showprogress for i ∈ 1:numCarves
        originalSize = size(seamCarver.image)
        newImage = zeros(RGB{N0f8}, (originalSize[1], originalSize[2]-1))
        newEnergy = zeros(Float64, (originalSize[1], originalSize[2]-1))
        minIdx = argmin(energy[originalSize[1], :])
        for j ∈ 1:originalSize[1]
            @inbounds newRow = seamCarver.image[originalSize[1] + 1 - j, 1:end .!= minIdx]
            # @inbounds newRow = my_hcat(newRow, seamCarver.image[originalSize[1] + 1 - j, minIdx+1:originalSize[2]])

            newImage[originalSize[1] + 1 - j, :] = newRow

            @inbounds newEnergyRow = energy[originalSize[1] + 1 - j, 1:end .!= minIdx]
            # @inbounds newEnergyRow = my_hcat(newEnergyRow, energy[originalSize[1] + 1 - j, minIdx+1:originalSize[2]])
            newEnergy[originalSize[1] + 1 - j, :] = newEnergyRow

            if j < originalSize[1]
                possibilities = [
                    minIdx - 1,
                    minIdx,
                    minIdx + 1
                ]

                if minIdx == 1; possibilities = possibilities[2:3]; end
                if minIdx == originalSize[2]; possibilities = possibilities[1:2]; end
                minIdx = sort(possibilities, by = (x -> energy[originalSize[1] - j, x]))[1]
            end
        end

        seamCarver.image  = newImage
        energy = newEnergy
    end
end

function paint_horizontal(seamCarver :: SeamCarver;
               numCarves :: Int64 = 1)

    energy = seamCarver.energyMapDict[:horizontal]
    canvas = deepcopy(seamCarver.image)
    @showprogress for i ∈ 1:numCarves
        originalSize = size(seamCarver.image)
        newEnergy = deepcopy(energy)
        minIdx = argmin(energy[originalSize[1], :])
        for j ∈ 1:originalSize[1]

            canvas[originalSize[1] + 1 - j, minIdx] = RGB(1.0/i, 0.0, 0.0)
            newEnergy[originalSize[1] + 1 - j, minIdx] = 1000

            if j < originalSize[1]
                possibilities = [
                    minIdx - 1,
                    minIdx,
                    minIdx + 1
                ]

                if minIdx == 1; possibilities = possibilities[2:3]; end
                if minIdx == originalSize[2]; possibilities = possibilities[1:2]; end
                minIdx = sort(possibilities, by = (x -> energy[originalSize[1] - j, x]))[1]
            end
        end

        energy = newEnergy
    end
    return canvas
end

function Base.show(io :: IO, seamCarver :: SeamCarver)
    display(seamCarver.image)
end


end
