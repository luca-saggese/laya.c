#ifndef HD_JSON_H
#define HD_JSON_H

/* Minimal recursive-descent JSON parser used by the M1 native loader.
 * Supports objects, arrays, strings, numbers (parsed as double + int64),
 * booleans and null. Complete handling of the standard escape sequences
 * required by our configs/manifests.
 */

#include <stddef.h>
#include <stdint.h>

typedef enum {
    HD_JSON_NULL = 0,
    HD_JSON_BOOL,
    HD_JSON_INT,
    HD_JSON_DOUBLE,
    HD_JSON_STRING,
    HD_JSON_ARRAY,
    HD_JSON_OBJECT,
} hd_json_type;

typedef struct hd_json hd_json;

typedef struct {
    hd_json **items;
    size_t count;
    size_t cap;
} hd_json_array;

typedef struct {
    char **keys;
    hd_json **values;
    size_t count;
    size_t cap;
} hd_json_object;

struct hd_json {
    hd_json_type type;
    union {
        int boolean;
        int64_t integer;
        double number;
        char *string;
        hd_json_array array;
        hd_json_object object;
    } u;
};

hd_json *hd_json_parse(const char *text, const char **err);
void hd_json_free(hd_json *v);

const hd_json *hd_json_get(const hd_json *obj, const char *key);
const char *hd_json_string(const hd_json *v);
int64_t hd_json_int(const hd_json *v, int64_t dflt);
double hd_json_double(const hd_json *v, double dflt);
int hd_json_bool(const hd_json *v, int dflt);
size_t hd_json_array_len(const hd_json *v);
const hd_json *hd_json_array_at(const hd_json *v, size_t i);

#endif /* HD_JSON_H */
