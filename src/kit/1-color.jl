export black!, red!, green!, yellow!, blue!, magenta!, cyan!, white!
export color

COLORS = Dict(
"black"         => 0,
"red"           => 1,
"green"         => 2,
"yellow"        => 3,
"blue"          => 4,
"magenta"       => 5,
"cyan"          => 6,
"white"         => 7,
"default"       => 9
)


MODES = Dict(
"default"       => 0,
"bold"          => 1,
"underline"     => 4,
"blink"         => 5,
"swap"          => 7,
"hide"          => 8
)


function color(color::String, str::String; background::String="default", mode::String="default")
    FColor = "\e[$(30 + COLORS[color])m"
    BColor = "\e[$(40 + COLORS[background])m"
    Action = "\e[$(MODES[mode])m"
    return Action * BColor * FColor * str * "\e[0m"
end


black!(str; mode::String="default")   = color("black",   string(str), mode=mode)
red!(str; mode::String="default")     = color("red",     string(str), mode=mode)
green!(str; mode::String="default")   = color("green",   string(str), mode=mode)
yellow!(str; mode::String="default")  = color("yellow",  string(str), mode=mode)
blue!(str; mode::String="default")    = color("blue",    string(str), mode=mode)
magenta!(str; mode::String="default") = color("magenta", string(str), mode=mode)
cyan!(str; mode::String="default")    = color("cyan",    string(str), mode=mode)
white!(str; mode::String="default")   = color("white",   string(str), mode=mode)
