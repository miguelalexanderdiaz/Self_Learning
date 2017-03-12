class Canvas {
    constructor(width, height) {
        this.svg = d3.select('body').append('svg').attr("width", width).attr("height", height);
    }
    add_square(width, height, color) {
        d3.select('svg').append('rect')
            .attr("width", width)
            .attr("height", height)
            .attr("fill", color);
    }
    rotate_object(obj, degrees) {
        d3.select(obj).attr("transform", 'rotate(' + degrees + ')');
    }
    translate_object(obj, x, y) {
        d3.select(obj).attr("transform", 'translate(' + x + ',' + y + ')');
        console.log(d3.select('svg.childNode'));
    }
}
let root = new Canvas(300, 400);
root.add_square(100, 150, "red");
root.rotate_object('rect', 20);
root.translate_object('rect', 100, 20);
//# sourceMappingURL=test.js.map